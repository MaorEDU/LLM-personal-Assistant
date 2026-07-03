# eval_device.py — device-level evaluation of the trained model on the REAL /ww
# recordings, simulating the firmware's streaming pipeline exactly:
#   rolling 1 s window, WW_HOP=100 ms, peak-normalize to WW_TARGET_PEAK (gain
#   capped at WW_MAX_GAIN, never attenuate), noise gate WW_MIN_PEAK on the raw
#   (codec-boosted) peak, fire = WW_CONSEC consecutive over-threshold windows.
# The recordings were captured WITHOUT the +18 dB codec listen boost, so the gate
# is evaluated on clip*BOOST; with the CMN front-end the SCORE itself is
# level-invariant, so this only affects gating, not model output.
#
# Run from wake_word_training/ after train.py:  python eval_device.py
import glob
import os
import numpy as np
import soundfile as sf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf
import feat

WW_DIR = os.environ.get("WW_DIR", "/home/user/Desktop/iot/ww")
HOP = 1600                 # WW_HOP
TARGET_PEAK = 0.60         # WW_TARGET_PEAK
MAX_GAIN = 25.0            # WW_MAX_GAIN
MIN_PEAK = 0.040           # WW_MIN_PEAK (on boosted signal)
CONSEC = int(os.environ.get("WW_EVAL_CONSEC", "2"))   # WW_CONSEC
BOOST = 10 ** (18 / 20)    # codec listen boost the device now applies (~7.94x)
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)

m = tf.keras.models.load_model("/tmp/ww/model.keras")
nz = np.load("/tmp/ww/norm.npz")
MEAN, STD = float(nz["mean"]), float(nz["std"])


def device_windows(a):
    """Streaming 1 s windows (hop 100 ms) exactly as the firmware sees them,
    including the fill-in phase where the phrase enters the rolling window."""
    a = np.concatenate([np.zeros(feat.CLIP - HOP, np.float32), a.astype(np.float32)])
    if len(a) < feat.CLIP:
        a = np.pad(a, (0, feat.CLIP - len(a)))
    n = (len(a) - feat.CLIP) // HOP + 1
    return np.stack([a[i * HOP:i * HOP + feat.CLIP] for i in range(n)])


def score_windows(wins):
    """ww_normalize + logmel(+CMN in feat) + model. Returns (probs1, gated_mask)."""
    peaks = np.abs(wins).max(axis=1)
    gains = np.clip(TARGET_PEAK / (peaks + 1e-9), 1.0, MAX_GAIN)
    normed = wins * gains[:, None]
    X = np.stack([feat.logmel(w) for w in normed])
    X = ((X - MEAN) / STD)[..., None]
    p1 = m.predict(X, verbose=0, batch_size=256)[:, 1]
    gated = (peaks * BOOST) < MIN_PEAK        # True = firmware squelches this window
    return p1, gated


def fires(p1, gated, thr):
    """Count debounced fire events (CONSEC consecutive scoring windows >= thr)."""
    run, events = 0, 0
    for s, g in zip(p1, gated):
        if g or s < thr:
            run = 0
            continue
        run += 1
        if run == CONSEC:
            events += 1
            run = 0
    return events


# ── positives: per-file detection (a miss = a child ignored) ──────────────────
pos_files = sorted(glob.glob(f"{WW_DIR}/pos/p*.wav"))
pos_scores = []            # per-file: debounced peak (min over CONSEC consecutive)
for p in pos_files:
    a, sr = sf.read(p, dtype="float32")
    p1, gated = score_windows(device_windows(a))
    eff = np.where(gated, -1.0, p1)
    if len(eff) >= CONSEC:
        runs = np.stack([eff[i:len(eff) - CONSEC + 1 + i] for i in range(CONSEC)])
        score = float(runs.min(axis=0).max())
    else:
        score = float(eff.max())
    if score < 0:
        print(f"  [note] {os.path.basename(p)}: every window gated "
              f"(raw peak {np.abs(a).max():.4f}) — bad capture, not a model miss")
    pos_scores.append(score)
pos_scores = np.array(pos_scores)

print(f"\nPOSITIVES ({len(pos_files)} real 'hey pip' files) — debounced peak score:")
print(f"  min={pos_scores.min():.3f}  p5={np.percentile(pos_scores,5):.3f}  "
      f"median={np.median(pos_scores):.3f}")
for thr in THRESHOLDS:
    print(f"  thr {thr:.2f}: recall {(pos_scores >= thr).mean():.3f} "
          f"({(pos_scores < thr).sum()} missed)")

# ── ambient: false fires over ~10 min of real room noise ─────────────────────
amb_files = sorted(glob.glob(f"{WW_DIR}/neg/c*.wav"))
amb_secs, amb_events = 0.0, {t: 0 for t in THRESHOLDS}
amb_events_ungated = {t: 0 for t in THRESHOLDS}
for p in amb_files:
    a, sr = sf.read(p, dtype="float32")
    amb_secs += len(a) / sr
    p1, gated = score_windows(device_windows(a))
    for t in THRESHOLDS:
        amb_events[t] += fires(p1, gated, t)
        amb_events_ungated[t] += fires(p1, np.zeros_like(gated), t)

print(f"\nAMBIENT ({len(amb_files)} files, {amb_secs/60:.1f} min) — debounced false fires:")
for t in THRESHOLDS:
    print(f"  thr {t:.2f}: {amb_events[t]} fires "
          f"({amb_events_ungated[t]} if the noise gate were off)")

# ── other speech: near-miss robustness ────────────────────────────────────────
neg_files = sorted(glob.glob(f"{WW_DIR}/neg/n*.wav"))
neg_fire = {t: 0 for t in THRESHOLDS}
offenders = []
for p in neg_files:
    a, sr = sf.read(p, dtype="float32")
    p1, gated = score_windows(device_windows(a))
    for t in THRESHOLDS:
        neg_fire[t] += 1 if fires(p1, gated, t) else 0
    if fires(p1, gated, 0.85):
        offenders.append(os.path.basename(p))

print(f"\nOTHER SPEECH ({len(neg_files)} files) — files that would false-fire:")
for t in THRESHOLDS:
    print(f"  thr {t:.2f}: {neg_fire[t]} / {len(neg_files)}")
if offenders:
    print("  offenders @0.85:", " ".join(offenders))

print("\nPick WAKE_WORD_THRESHOLD = highest thr with recall ~1.00 and ~0 ambient fires.")
