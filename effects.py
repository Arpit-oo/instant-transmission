"""Compositing + Instant Transmission visual effects.

Everything stays uint8 and uses OpenCV native ops (blendLinear, addWeighted)
so a 720p frame composites in a few milliseconds.
"""
import cv2
import numpy as np

GLOW_BGR = (255, 235, 140)  # cyan-white ki glow


class FX:
    def __init__(self, size, glow=False):
        self.w, self.h = size
        self.glow = glow
        self.solid = np.empty((self.h, self.w, 3), np.uint8)
        self.solid[:] = GLOW_BGR
        self.rng = np.random.default_rng()

    # ---- basics -------------------------------------------------------------
    @staticmethod
    def over(base, layer, alpha):
        """base * (1 - alpha) + layer * alpha, alpha float32 HxW in [0,1]."""
        return cv2.blendLinear(base, layer, 1.0 - alpha, alpha)

    def remove_person(self, frame, plate_u8, removal):
        return self.over(frame, plate_u8, removal)

    def shift(self, img, alpha, dx):
        if dx == 0:
            return img, alpha
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (self.w, self.h), borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(alpha, M, (self.w, self.h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img, alpha

    # ---- effects ------------------------------------------------------------
    def glitch(self, img, alpha, intensity):
        """Horizontal slice tearing + RGB split on the person layer."""
        if intensity <= 0.01:
            return img, alpha
        rng = self.rng
        img = img.copy()
        alpha = alpha.copy()
        for _ in range(int(3 + 10 * intensity)):
            y = int(rng.integers(0, self.h))
            bh = int(rng.integers(3, max(4, int(self.h * 0.05))))
            dx = int(rng.normal(0, 45 * intensity))
            if dx:
                img[y:y + bh] = np.roll(img[y:y + bh], dx, axis=1)
                alpha[y:y + bh] = np.roll(alpha[y:y + bh], dx, axis=1)
        k = int(14 * intensity)
        if k:
            b, g, r = cv2.split(img)
            Mr = np.float32([[1, 0, k], [0, 1, 0]])
            Mb = np.float32([[1, 0, -k], [0, 1, 0]])
            r = cv2.warpAffine(r, Mr, (self.w, self.h), borderMode=cv2.BORDER_REPLICATE)
            b = cv2.warpAffine(b, Mb, (self.w, self.h), borderMode=cv2.BORDER_REPLICATE)
            img = cv2.merge((b, g, r))
        return img, alpha

    def rim_glow(self, out, alpha, intensity):
        """Add a bright outline around the silhouette (computed at 1/4 res)."""
        if intensity <= 0.01:
            return out
        sw, sh = self.w // 4, self.h // 4
        a = cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_AREA)
        edge = cv2.absdiff(cv2.GaussianBlur(a, (0, 0), 2.0), a)
        edge = np.clip(edge * (4.0 * intensity), 0, 1)
        glow = cv2.merge([(edge * c).astype(np.uint8) for c in GLOW_BGR])
        glow = cv2.resize(glow, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return cv2.add(out, glow)

    def transition(self, base, layer, alpha, t, appearing=False):
        """t in [0,1]. Vanish: person flickers out with tearing + glow. Appear: reverse."""
        t = min(1.0, max(0.0, float(t)))
        vis_t = t if appearing else 1 - t  # 1 = fully visible
        intensity = max(0.0, float(np.sin(np.pi * t))) ** 0.7
        flicker = 1.0 if self.rng.random() > intensity * 0.6 else 0.0
        vis = (vis_t ** 0.8) * flicker
        if appearing and t > 0.85:
            vis = vis_t  # land solidly

        g_img, g_alpha = self.glitch(layer, alpha, intensity)
        out = self.over(base, g_img, g_alpha * np.float32(vis))
        if self.glow:
            out = self.rim_glow(out, alpha, intensity)
        flash = max(0.0, 1 - abs(t - 0.5) / 0.12) * 0.25  # ki flash at midpoint
        if flash > 0:
            out = cv2.addWeighted(out, 1 - flash, self.solid, flash, 0)
        return out

    def make_ghost(self, layer, alpha):
        tint = cv2.addWeighted(layer, 0.6, self.solid, 0.4, 0)
        return tint, alpha.copy()

    def draw_ghost(self, out, ghost, u):
        """Fading afterimage, u in [0,1] (0 = just vanished)."""
        tint, alpha = ghost
        return self.over(out, tint, alpha * np.float32(0.55 * (1 - u) ** 2))
