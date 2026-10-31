"""Synthesized Instant Transmission sound effects (no asset files needed)."""
import os
import wave

import numpy as np

try:
    import winsound
except ImportError:  # non-Windows
    winsound = None

SR = 44100
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _sweep(f0, f1, dur, noise=0.35):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    freq = f0 * (f1 / f0) ** (t / dur)  # exponential sweep
    phase = 2 * np.pi * np.cumsum(freq) / SR
    tone = np.sin(phase) + 0.4 * np.sin(2 * phase)
    hiss = np.random.default_rng(7).normal(0, 1, t.size)
    env = np.exp(-t / (dur * 0.35)) * np.minimum(1, t / 0.004)
    sig = (tone * (1 - noise) + hiss * noise) * env
    return sig / np.max(np.abs(sig)) * 0.8


def _write(path, sig):
    data = (sig * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(data.tobytes())


class Sfx:
    def __init__(self, enabled=True):
        self.enabled = enabled and winsound is not None
        os.makedirs(ASSET_DIR, exist_ok=True)
        self.vanish = os.path.join(ASSET_DIR, "vanish.wav")
        self.appear = os.path.join(ASSET_DIR, "appear.wav")
        if not os.path.exists(self.vanish):
            _write(self.vanish, _sweep(2400, 180, 0.22))
        if not os.path.exists(self.appear):
            _write(self.appear, _sweep(180, 2400, 0.22))

    def play(self, path):
        if self.enabled:
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
