"""Instant Transmission gesture detection: index + middle finger on forehead."""
import math

import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

# Face mesh landmarks
FOREHEAD_TOP = 10
FOREHEAD_LOW = 9  # just above the brows
FACE_LEFT, FACE_RIGHT = 234, 454

# Hand landmarks (tip, pip)
WRIST = 0
INDEX = (8, 6)
MIDDLE = (12, 10)
RING = (16, 14)
PINKY = (20, 18)


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seg_dist(p, a, b):
    """Distance from point p to segment a-b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return _d(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return _d(p, (ax + t * dx, ay + t * dy))


class GestureDetector:
    def __init__(self, hold_s=0.3, cooldown_s=1.0, face_memory_s=1.0):
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.face = mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hold_s = hold_s
        self.cooldown_s = cooldown_s
        self.face_memory_s = face_memory_s

        self._face = None  # (top, low, face_width) in px
        self._face_t = -1e9
        self._pose_since = None
        self._last_true = -1e9
        self._last_fire = -1e9
        self._armed = True
        self.debug = {}

    def close(self):
        self.hands.close()
        self.face.close()

    def _update_face(self, rgb, now, w, h):
        res = self.face.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            px = lambda i: (lm[i].x * w, lm[i].y * h)
            fw = _d(px(FACE_LEFT), px(FACE_RIGHT))
            self._face = (px(FOREHEAD_TOP), px(FOREHEAD_LOW), max(fw, 1.0))
            self._face_t = now
        elif now - self._face_t > self.face_memory_s:
            self._face = None
        return self._face

    def _hand_pose(self, pts, face):
        top, low, fw = face
        wrist = pts[WRIST]

        def ratio(f):
            tip, pip = f
            return _d(wrist, pts[tip]) / max(_d(wrist, pts[pip]), 1e-6)

        r_idx, r_mid, r_ring, r_pky = ratio(INDEX), ratio(MIDDLE), ratio(RING), ratio(PINKY)
        two_up = r_idx > 1.2 and r_mid > 1.2
        others_down = r_ring < 1.15 and r_pky < 1.15
        together = _d(pts[INDEX[0]], pts[MIDDLE[0]]) < 0.45 * fw
        tip_dist = min(_seg_dist(pts[INDEX[0]], top, low), _seg_dist(pts[MIDDLE[0]], top, low))
        on_forehead = tip_dist < 0.4 * fw
        return two_up and others_down and together and on_forehead, {
            "two_up": two_up,
            "others_down": others_down,
            "together": together,
            "on_forehead": on_forehead,
            "tips": (pts[INDEX[0]], pts[MIDDLE[0]]),
        }

    def update(self, rgb, now):
        """Feed an RGB frame. Returns (fired, progress 0..1)."""
        h, w = rgb.shape[:2]
        pose = False
        face = None
        self.debug = {"face": None, "hands": []}

        # Hands first; face mesh only runs when a hand is in view (saves CPU).
        res = self.hands.process(rgb)
        if res.multi_hand_landmarks:
            face = self._update_face(rgb, now, w, h)
            self.debug["face"] = face
            if face is not None:
                for hand in res.multi_hand_landmarks:
                    pts = [(p.x * w, p.y * h) for p in hand.landmark]
                    ok, info = self._hand_pose(pts, face)
                    self.debug["hands"].append(info)
                    pose = pose or ok

        # Debounce: tolerate tracking dropouts shorter than 120ms.
        if pose:
            self._last_true = now
            if self._pose_since is None:
                self._pose_since = now
        elif now - self._last_true > 0.12:
            self._pose_since = None
            if now - self._last_true > 0.25:
                self._armed = True

        progress = 0.0
        fired = False
        if self._pose_since is not None and self._armed:
            progress = min(1.0, (now - self._pose_since) / self.hold_s)
            if progress >= 1.0 and now - self._last_fire > self.cooldown_s:
                fired = True
                self._armed = False
                self._last_fire = now
        self.debug["pose"] = pose
        return fired, progress
