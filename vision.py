"""Camera capture, person segmentation, and clean-plate background model."""
import threading
import time

import cv2
import mediapipe as mp
import numpy as np


class Camera:
    """Threaded webcam reader that always hands out the latest frame."""

    def __init__(self, index=0, width=1280, height=720, fps=30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Could not open camera {index}. Is another app using it?")
        self.size = (frame.shape[1], frame.shape[0])
        self.fps = fps
        self._frame = frame
        self._seq = 1
        self._read_seq = 0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._seq += 1
            else:
                time.sleep(0.01)

    def read(self, timeout=1.0):
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if self._seq != self._read_seq:
                    self._read_seq = self._seq
                    return self._frame.copy()
            time.sleep(0.001)
        with self._lock:
            return self._frame.copy()

    def close(self):
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()


class Segmenter:
    """Person mask via MediaPipe Selfie Segmentation.

    Returns three full-res masks:
      soft     float32 - tight, smoothed alpha of the person (for drawing them)
      removal  float32 - generous erase mask: low threshold, union of the last
                         few frames (covers motion lag), dilated, feathered
                         OUTWARD so the silhouette edge is fully erased
      learn_ok uint8   - 255 where it is safe to learn background (far from you)
    """

    HISTORY = 4  # frames of mask history unioned into the erase mask (~130ms @30fps)

    def __init__(self, full_size):
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.full_size = full_size
        self._prev = None
        self._hist = []
        self._centroid = None
        self._v = (0.0, 0.0)
        self.person_frac = 0.0
        self._k_erase = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        self._k_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))

    def close(self):
        self.seg.close()

    def _velocity(self, core):
        """Smoothed centroid velocity (px/frame at processing res), None if no person."""
        mo = cv2.moments(core, binaryImage=True)
        if mo["m00"] < 50:
            self._centroid = None
            return None
        c = (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"])
        prev = self._centroid
        self._centroid = c
        if prev is None:
            return None
        v = (c[0] - prev[0], c[1] - prev[1])
        sv = self._v
        self._v = (0.5 * v[0] + 0.5 * sv[0], 0.5 * v[1] + 0.5 * sv[1])
        return self._v

    def process(self, rgb_small):
        raw = self.seg.process(rgb_small).segmentation_mask.astype(np.float32)

        # Drawing mask: smooth, tight.
        m = np.clip((raw - 0.25) / 0.5, 0.0, 1.0)
        self._prev = m if self._prev is None else 0.55 * m + 0.45 * self._prev
        soft = cv2.GaussianBlur(self._prev, (5, 5), 0)

        # Erase mask: anything remotely person-like, over the last few frames.
        core = (raw > 0.08).astype(np.uint8)
        self.person_frac = float(cv2.countNonZero(core)) / core.size
        self._hist.append(core)
        self._hist = self._hist[-self.HISTORY:]
        union = self._hist[0].copy()
        for c in self._hist[1:]:
            cv2.bitwise_or(union, c, dst=union)
        # Segmentation lags real motion: the leading edge of a moving person
        # sits ahead of the mask. Predict it by pushing the mask forward along
        # the centroid velocity.
        v = self._velocity(core)
        if v is not None:
            vx, vy = v
            for k in (1.5, 3.0):
                M = np.float32([[1, 0, vx * k], [0, 1, vy * k]])
                ahead = cv2.warpAffine(core, M, (core.shape[1], core.shape[0]))
                cv2.bitwise_or(union, ahead, dst=union)
        grown = cv2.dilate(union, self._k_erase).astype(np.float32)
        removal = np.clip(cv2.GaussianBlur(grown, (0, 0), 5) * 2.0, 0.0, 1.0)

        # Background learning only well away from the person.
        learn_ok = (cv2.dilate(union, self._k_safe) == 0).astype(np.uint8) * 255

        fs = self.full_size
        soft = cv2.resize(soft, fs, interpolation=cv2.INTER_LINEAR)
        removal = cv2.resize(removal, fs, interpolation=cv2.INTER_LINEAR)
        learn_ok = cv2.resize(learn_ok, fs, interpolation=cv2.INTER_NEAREST)
        return soft, removal, learn_ok


class CleanPlate:
    """Background model of the room WITHOUT you in it. It must never contain you.

    - Explicit capture waits until the segmenter sees an empty room, then takes
      a median of N frames. (A blind countdown baked people into the plate.)
    - If capture is skipped, pixels are only ever taken from frames where the
      segmenter says no person is anywhere near them. Unseen pixels are
      inpainted from their surroundings.
    - Ongoing refresh only touches pixels that already match the plate, so a
      person (or anything new) can never be absorbed. Exposure drift is
      handled separately by matched().
    """

    EMPTY_FRAC = 0.003  # person pixels (fraction of frame) still counted as "empty room"
    EMPTY_HOLD_S = 1.0  # room must stay empty this long before capturing

    def __init__(self, size, capture_frames=15, learn_rate=0.1, learn_every=4, match_tol=20):
        self.w, self.h = size
        self.capture_frames = capture_frames
        self.learn_rate = learn_rate
        self.learn_every = learn_every
        self.match_tol = match_tol
        self.img = None  # float32 BGR
        self.known = None  # uint8 0/255
        self.capturing = False
        self._empty_since = None
        self._frames = []
        self._gain = None
        self._sigma = 0.0
        self._noise = None
        self._rng = np.random.default_rng()
        self._p_small = None
        self._calls = 0
        self._updates = 0

    # ---- explicit capture -------------------------------------------------
    def start_capture(self, now):
        self.capturing = True
        self._empty_since = None
        self._frames = []

    def skip_capture(self):
        self.capturing = False
        self._frames = []

    def tick_capture(self, frame, now, person_frac):
        """Returns a status string while capturing, None when done."""
        if person_frac > self.EMPTY_FRAC:
            self._empty_since = None
            self._frames = []
            return "STEP OUT OF FRAME  (SPACE to skip)"
        if self._empty_since is None:
            self._empty_since = now
        if now - self._empty_since < self.EMPTY_HOLD_S:
            return "ROOM EMPTY - HOLD STILL..."
        self._frames.append(frame)
        if len(self._frames) < self.capture_frames:
            return "CAPTURING BACKGROUND..."
        self.img = np.median(np.stack(self._frames), axis=0).astype(np.float32)
        self.known = np.full((self.h, self.w), 255, np.uint8)
        self._p_small = None
        self._frames = []
        self.capturing = False
        return None

    # ---- continuous learning ----------------------------------------------
    def update(self, frame, learn_ok):
        """learn_ok: uint8 mask, 255 where no person is anywhere near."""
        if self.img is None:
            self.img = np.zeros((self.h, self.w, 3), np.float32)
            self.known = np.zeros((self.h, self.w), np.uint8)
        self._updates += 1

        # 1. First sighting of a pixel with nobody near it: take it as-is.
        new = cv2.bitwise_and(learn_ok, cv2.bitwise_not(self.known))
        if cv2.countNonZero(new):
            self.img[new > 0] = frame[new > 0]
            cv2.bitwise_or(self.known, new, dst=self.known)
            self._p_small = None

        # 2. Gentle refresh, only where the live pixel already matches the plate.
        if self._updates % self.learn_every == 0:
            close = cv2.absdiff(frame, cv2.convertScaleAbs(self.img)).max(axis=2) < self.match_tol
            ok = cv2.bitwise_and(learn_ok, close.astype(np.uint8) * 255)
            cv2.accumulateWeighted(frame, self.img, self.learn_rate, mask=ok)

    def fill_unknown(self):
        """Inpaint regions never seen without a person in front of them."""
        if self.img is None or self.known.min() == 255:
            return
        sw, sh = self.w // 8, self.h // 8
        small = cv2.resize(self.img, (sw, sh)).astype(np.uint8)
        hole = cv2.resize(255 - self.known, (sw, sh), interpolation=cv2.INTER_NEAREST)
        hole = cv2.dilate(hole, np.ones((3, 3), np.uint8))
        filled = cv2.inpaint(small, hole, 5, cv2.INPAINT_TELEA)
        filled = cv2.resize(filled, (self.w, self.h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        unknown = self.known == 0
        self.img[unknown] = filled[unknown]
        self._p_small = None

    def u8(self):
        return cv2.convertScaleAbs(self.img)

    def matched(self, frame, learn_ok, removal):
        """Plate adjusted to look like the *live* camera right now.

        Webcams constantly change exposure / white balance and add sensor
        grain. A frozen plate therefore shows up as a slightly-off silhouette
        where you were. Fix:
          1. Local gain map (per channel) from live vs plate in the visible
             background, interpolated across the area you cover.
          2. Fresh grain matching the camera's measured noise level.
        """
        gw, gh = 64, 36
        f = cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)
        # plate changes slowly: refresh its low-res copy every 10 calls
        self._calls += 1
        if self._p_small is None or self._calls % 10 == 0:
            self._p_small = cv2.resize(self.img, (gw, gh), interpolation=cv2.INTER_AREA)
        p = self._p_small
        m = (cv2.resize(learn_ok, (gw, gh), interpolation=cv2.INTER_AREA) > 250).astype(np.float32)

        if m.sum() >= 20:
            m3 = m[..., None]
            g_global = ((f * m3).sum((0, 1)) + 1) / ((p * m3).sum((0, 1)) + 1)
            blur = lambda x: cv2.GaussianBlur(x, (0, 0), 6)
            den = blur(m)[..., None]
            local = (blur(f * m3) + 1) / (blur(p * m3) + 1)
            w = den / (den + 0.05)  # trust local estimate where background is visible nearby
            gain = w * local + (1 - w) * g_global
            gain = np.clip(gain, 0.5, 2.0).astype(np.float32)
            self._gain = gain if self._gain is None else 0.6 * gain + 0.4 * self._gain
        gain = self._gain if self._gain is not None else np.ones((gh, gw, 3), np.float32)

        # Camera grain: robust noise estimate (MAD) on sampled background pixels.
        sel = learn_ok[::8, ::8] > 0
        if sel.sum() > 200:
            ps = self.img[::8, ::8]
            gs = cv2.resize(gain, (ps.shape[1], ps.shape[0]), interpolation=cv2.INTER_LINEAR)
            diff = (frame[::8, ::8].astype(np.float32) - ps * gs)[sel]
            diff -= np.median(diff, axis=0)
            sigma = float(np.median(np.abs(diff)) * 1.4826)
            self._sigma = 0.8 * self._sigma + 0.2 * min(sigma, 8.0)
        if self._noise is None:  # pool of unit-variance grain frames, generated once
            self._noise = []
            for _ in range(6):
                n = np.empty((self.h, self.w, 3), np.float32)
                cv2.randn(n, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
                self._noise.append(n)

        # Only the area being erased needs the plate; do the full-res work there.
        out = frame.copy()
        x, y, bw, bh = cv2.boundingRect((removal > 0.002).astype(np.uint8))
        if bw == 0 or bh == 0:
            return out
        g = cv2.resize(gain, (self.w, self.h), interpolation=cv2.INTER_LINEAR)[y:y + bh, x:x + bw]             if bw * bh > 0.6 * self.w * self.h else self._crop_gain(gain, x, y, bw, bh)
        roi = cv2.multiply(self.img[y:y + bh, x:x + bw], g)
        if self._sigma > 0.5:
            n = self._noise[self._rng.integers(len(self._noise))]
            roi = cv2.scaleAdd(n[y:y + bh, x:x + bw], self._sigma, roi)
        out[y:y + bh, x:x + bw] = cv2.convertScaleAbs(roi)
        return out

    def _crop_gain(self, gain, x, y, bw, bh):
        """Upsample only the needed part of the low-res gain map (same result as full resize + crop)."""
        gh, gw = gain.shape[:2]
        sx, sy = gw / self.w, gh / self.h
        mx = (np.arange(x, x + bw, dtype=np.float32) + 0.5) * sx - 0.5
        my = (np.arange(y, y + bh, dtype=np.float32) + 0.5) * sy - 0.5
        map_x = np.broadcast_to(mx[None, :], (bh, bw)).astype(np.float32)
        map_y = np.broadcast_to(my[:, None], (bh, bw)).astype(np.float32)
        return cv2.remap(gain, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    @property
    def coverage(self):
        return 0.0 if self.known is None else float(np.count_nonzero(self.known)) / self.known.size
