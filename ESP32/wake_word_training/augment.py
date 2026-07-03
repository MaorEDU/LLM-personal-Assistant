# augment.py — clip-level audio augmentation for the "hey pip" trainer.
# Turns one clean utterance into a realistic 1 s clip (peak in [0.25,1.0], so it
# brackets the device's runtime peak-normalize target of 0.60 — see wake_word.h).
# Pure numpy/scipy + feat (no librosa here, so it stays fast); an optional
# `pitch_shifter` callable can be injected by build_dataset.py for kid/woman pitch.
import numpy as np
from scipy.signal import fftconvolve, butter, lfilter
import feat

SR, CLIP = feat.SR, feat.CLIP


def _rir():
    """Random short exponential room impulse response."""
    L = np.random.randint(200, 1600)
    ir = np.exp(-np.arange(L) / np.random.uniform(40, 300))
    ir *= (np.random.rand(L) < 0.5)
    ir[0] = 1.0
    return ir.astype(np.float32)


def _bandlimit(x):
    """Mimic the partly-covered mic / tiny speaker: random low-pass + mild high-pass.
    Low-pass floor kept at 3.5 kHz (not 3 kHz): the real recordings ALREADY went
    through the covered mic, so over-filtering here erases the plosive cues that
    separate 'pip' from 'pick'/'pin'."""
    lo = np.random.uniform(3500, 7800)
    b, a = butter(4, lo / (SR / 2), btype="low")
    x = lfilter(b, a, x)
    if np.random.rand() < 0.5:
        b, a = butter(2, 120 / (SR / 2), btype="high")
        x = lfilter(b, a, x)
    return x.astype(np.float32)


def _varispeed(u):
    """Cheap speed+pitch jitter via resample (no librosa)."""
    f = np.random.uniform(0.9, 1.1)
    n = max(8, int(len(u) / f))
    return np.interp(np.linspace(0, len(u) - 1, n), np.arange(len(u)), u).astype(np.float32)


def place_in_clip(u):
    """Drop the utterance at a random offset inside a 1 s buffer (time-shift aug)."""
    u = u[:CLIP - 100] if len(u) > CLIP - 100 else u
    buf = np.zeros(CLIP, np.float32)
    off = np.random.randint(0, max(1, CLIP - len(u)))
    buf[off:off + len(u)] += u
    return buf


def rms(x):
    return float(np.sqrt(np.mean(x * x)) + 1e-12)


def _fit(n):
    """Make a noise array exactly CLIP long."""
    if len(n) < CLIP:
        return np.pad(n, (0, CLIP - len(n)))
    return n[:CLIP]


def augment_speech(u, noise_pool, pitch_shifter=None):
    """u: clean utterance (float32). Returns a realistic 1 s clip (float32)."""
    s = u.astype(np.float32).copy()
    if np.random.rand() < 0.5:
        s = _varispeed(s)
    if pitch_shifter is not None and np.random.rand() < 0.20:
        try:
            s = pitch_shifter(s)
        except Exception:
            pass
    if np.random.rand() < 0.30:
        s = fftconvolve(s, _rir())[:len(s) + 200].astype(np.float32)
    s = _bandlimit(s)
    s = s / (np.max(np.abs(s)) + 1e-9) * np.random.uniform(0.10, 1.0)   # wide level range — CMN in
    # feat.logmel makes the model level-invariant, so this now only varies the
    # epsilon-floor/SNR regime rather than needing to bracket a device target level.
    buf = place_in_clip(s)
    if noise_pool is not None and len(noise_pool):
        n = _fit(noise_pool[np.random.randint(len(noise_pool))]).astype(np.float32)
        snr = np.random.uniform(6, 30)   # realistic room SNR (3 dB floor buried the phrase)
        buf = buf + n * (rms(buf) / (10 ** (snr / 20)) / (rms(n) + 1e-9))
    m = np.max(np.abs(buf))
    if m > 1:
        buf = buf / m
    return buf.astype(np.float32)
