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
    """Background model of the room without you in it.

    - Optional explicit capture (step out of frame, median of N frames).
    - Continuously refreshed wherever you are NOT, so it tracks lighting
      changes and fills in as you move around.
    - Never-seen pixels are inpainted when needed.
    """

    def __init__(self, size, countdown_s=3.0, capture_frames=15, learn_rate=0.03):
        self.w, self.h = size
        self.countdown_s = countdown_s
        self.capture_frames = capture_frames
        self.learn_rate = learn_rate
        self.img = None  # float32 BGR
        self.known = None  # uint8 0/255
        self.capturing = False
        self._cap_start = 0.0
        self._frames = []

    # ---- explicit capture -------------------------------------------------
    def start_capture(self, now):
        self.capturing = True
        self._cap_start = now
        self._frames = []

    def tick_capture(self, frame, now):
        """Returns a status string while capturing, None when done."""
        left = self.countdown_s - (now - self._cap_start)
        if left > 0:
            return f"STEP OUT OF FRAME  {int(left) + 1}"
        self._frames.append(frame)
        if len(self._frames) < self.capture_frames:
            return "CAPTURING BACKGROUND..."
        self.img = np.median(np.stack(self._frames), axis=0).astype(np.float32)
        self.known = np.full((self.h, self.w), 255, np.uint8)
        self._frames = []
        self.capturing = False
        return None

    # ---- continuous learning ----------------------------------------------
    def update(self, frame, learn_ok):
        """learn_ok: uint8 mask, 255 where no person is anywhere near."""
        bg = learn_ok
        if self.img is None:
            self.img = frame.astype(np.float32)
            self.known = bg.copy()
            return
        cv2.accumulateWeighted(frame, self.img, self.learn_rate, mask=bg)
        cv2.bitwise_or(self.known, bg, dst=self.known)

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

    def u8(self):
        return cv2.convertScaleAbs(self.img)

    @property
    def coverage(self):
        return 0.0 if self.known is None else float(np.count_nonzero(self.known)) / self.known.size
