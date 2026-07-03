# RETRAIN RUNBOOK — robust "hey pip" wake word

> **You are Claude running on an Ubuntu machine (a VMware VM).** This file is a
> self-contained handoff: everything you need is here — do **not** assume any prior
> chat history or memory. Read it top to bottom, then execute. Ask the user only if a
> step truly blocks you (e.g. missing recordings or no internet).

---

## 0. Mission & background

This repo is a kids' Hebrew homework-tutor on an **ESP32-S3** (LCDWIKI 2.8", ES8311
codec). It wakes on the phrase **"hey pip"** using a tiny on-device CNN (log-mel +
3-conv) — pure C, no TensorFlow at runtime. The model lives in
`ESP32/homework_assistant/hey_pip_model.h` and was trained by the pipeline in **this
folder** (`ESP32/wake_word_training/`).

**v1 was trained ONLY on synthetic espeak voices and it failed in the field:** a real
adult-male "hey pip" is *not* detected, while random bed/room noise *does* false-fire.
Root cause = domain gap (espeak ≠ real voices) + a weak noise class. Front-end tuning
(thresholds/gates) was already tried and is not enough.

**Your job:** retrain the model to be robust, using **(a) real audio the user recorded
through the actual device mic** + **(b) much better synthetic data** (neural TTS +
heavy augmentation + hard negatives). Produce a new `hey_pip_model.h`, prove C/Python
parity, and hand it back to drop into the firmware.

Why retrain instead of a turnkey engine: no off-the-shelf wake-word engine supports
Hebrew, and the robust ones (microWakeWord, ESP-SR/WakeNet) require ESP-IDF 5.x / Arduino
core 3.x, which would break this project's MP3 TTS path (it's pinned to Arduino core
2.0.17). So we keep the self-contained model and just make the **data** far better.

---

## 1. HARD INVARIANTS — the device contract (do NOT break these)

The runtime C inference (`ww_infer.h`) is **fixed**. Only `hey_pip_model.h` (weights +
constants) is regenerated. Therefore:

1. **Keep the front-end identical.** Do **not** change the constants in `feat.py`:
   `SR=16000, CLIP=16000, FRAME_LEN=400, FRAME_STEP=160, NFFT=512, N_MEL=40,
   FMIN=20, FMAX=8000` (→ 98 frames × 40 mels). `feat.logmel` must stay byte-equivalent
   to `ww_infer.h`'s `ww_logmel`. If you ever change these, you must also edit
   `ww_infer.h` on the device — avoid it.
2. **Keep the architecture topology.** Exactly: `Conv2D(c1,3,stride2,'valid',relu)` →
   `Conv2D(c2,3,stride2,'valid',relu)` → `Conv2D(c3,3,stride2,'valid',relu)` →
   `GlobalAveragePooling2D` → `Dense(3,'softmax')`. `ww_infer.h` hardcodes: 3 conv
   layers, kernel **3**, stride **2**, 'valid' padding, GAP, one dense, softmax.
   - You **MAY** freely change the channel counts `c1/c2/c3` (export writes
     `WW_C1_OUT/2/3`). Bumping capacity (e.g. 16/24/32 → 24/40/64) is encouraged for
     robustness.
   - You **MAY NOT** add/remove layers, change kernel size, change stride, or add
     batch-norm/dense-hidden layers without editing `ww_infer.h` (don't, unless asked).
3. **Keep 3 classes in this exact order: `0=noise, 1=hey-pip, 2=other-speech`.** The
   firmware reads `prob[1]` as the wake-word score. Do not reorder.
4. **Export MEAN/STD.** `ww_infer.h` applies `(feat-WW_MEAN)/WW_STD`. `train.py` computes
   these over the training set and `export_c.py` writes them. Keep that flow.
5. **The device pre-normalizes audio before inference.** At runtime `wake_word.h`
   scales each 1 s window so its **peak ≈ 0.60** (`WW_TARGET_PEAK`, capped) before
   calling the model. So your training clips must live at realistic peaks — keep level
   augmentation in the range **peak ∈ [0.25, 1.0]** (brackets 0.60). Peak-normalize the
   real recordings the same way.
6. **Gate at the end with `parity.py`** (Section 7). The C model must match Keras to
   `<1e-3` before you ship the header.

---

## 2. Get the files onto this machine

You need two things present locally:

**A) This repo (or at least `ESP32/wake_word_training/` + `ESP32/homework_assistant/`).**
Ask the user how they'll provide it. Likely one of:
- `git clone <their repo url>` (if it's on GitHub), or
- a VMware **shared folder** / drag-drop of the project folder, or
- `scp` from the Windows host.

**B) The real recordings** the user captured on the device SD card: a **`/ww`** folder:
```
ww/pos/pNNNN.wav      ← "hey pip" utterances            (label 1)
ww/neg/nNNNN.wav      ← short non-wake speech            (label 2)
ww/neg/cNNNN.wav      ← long continuous ambient/noise    (label 0)
```
All are **16 kHz mono 16-bit WAV**. Confirm with the user where they copied this folder
(e.g. `~/ww`). If it's missing, stop and ask — the real data is the whole point.

Sanity-check the recordings before training:
```bash
python3 - <<'PY'
import soundfile as sf, glob, numpy as np
for d,lab in [("pos",1),("neg/n*",2),("neg/c*",0)]:
    fs=sorted(glob.glob(f"$HOME/ww/{d}*.wav")) if "*" in d else sorted(glob.glob(f"$HOME/ww/{d}/*.wav"))
    if not fs: continue
    a,_=sf.read(fs[0]); print(d, "files:",len(fs), "sr-ok, dur0=%.2fs peak0=%.3f"%(len(a)/16000,np.max(np.abs(a))))
PY
```
Expect: positives ~1.8 s, ambient ~20 s, peaks well above noise. If positives are
near-silent or clipped, tell the user to re-record (closer / further).

---

## 3. Environment setup

This tiny CNN trains on CPU in minutes. TensorFlow works fine on Ubuntu (unlike the
user's Windows Python 3.13 — that's why training is here).

```bash
sudo apt-get update
sudo apt-get install -y espeak-ng ffmpeg        # espeak-ng: keeps parity.py/old tts.py working; ffmpeg: decode edge-tts mp3
python3 -m venv ~/wwenv && source ~/wwenv/bin/activate
pip install --upgrade pip
pip install "tensorflow-cpu" numpy scipy soundfile librosa edge-tts espeakng_loader py-espeak-ng
```
- **edge-tts needs internet** (Microsoft neural voices). Verify: `edge-tts --list-voices | grep -E "en-US|he-IL"`.
  You should see English voices (e.g. `en-US-AriaNeural`, `en-US-GuyNeural`) **and Hebrew**
  (`he-IL-AvriNeural`, `he-IL-HilaNeural`).
- If there's **no internet**, fall back to the existing espeak `tts.py` for synthetic
  voices (lower quality) and lean harder on the real recordings + augmentation.

Working dir convention used by the existing scripts: **`/tmp/ww`** (and `/tmp/ww/out`).
```bash
mkdir -p /tmp/ww/out
```

Read these files first so you understand the contract:
`feat.py`, `gen_data.py`, `train.py`, `export_c.py`, `parity.py`, `ww_infer.h`,
`host_test.c`, and on the device side `../homework_assistant/wake_word.h` (the runtime
normalization in §1.5) and `../homework_assistant/hey_pip_model.h` (current header format).

---

## 4. The plan (what "better data" means)

Build the training set from four sources, all passed through `feat.logmel` (unchanged):

| Source | Class | Notes |
|---|---|---|
| **Real "hey pip"** (`ww/pos`) | 1 | The anchor. Augment each ~30–50×. |
| **Neural-TTS "hey pip"** (edge-tts, EN + HE voices) | 1 | Diversity across speakers/accents. |
| **Real ambient** (`ww/neg/c*`, sliced to 1 s) | 0 | Kills the bed-noise false fire. |
| **Real other-speech** (`ww/neg/n*`) + **TTS hard negatives** | 2 | "hey", "pip", "hey kid", "pizza", kid answers, near-misses. |
| **Synthetic noise** (white/pink/brown/hum) | 0 | Existing `gen_data.rnoise`. |

Every clip is augmented with: **level** (peak→U[0.25,1.0]), **noise mix** at SNR
U[3,28] dB (mix in *real* ambient too, not just synthetic), **reverb** (random
exponential RIR), **random time offset** in the 1 s window, **vari-speed/pitch**
(resample ×U[0.9,1.1], and/or `librosa.effects.pitch_shift` ±3 semitones to simulate
kids/women), and a **band-limit/EQ** (random low-pass 3–7 kHz + mild high-pass) to mimic
the covered-mic / small-speaker response. SpecAugment (time/freq masking) at train time.

> **Spend augmentation on NEGATIVES too.** The reported failure is *false fires on
> noise*, so the model needs a rich, realistic class-0/2. Slice ALL the long ambient
> recordings into overlapping 1 s windows (hop 0.5 s) and include them generously.

---

## 5. Implement the pipeline

You have two valid approaches — pick based on your judgment:

**Approach A (recommended): extend the existing scripts.** Keep `feat.py` and
`export_c.py` as-is. Add a `real_data.py` (loads `ww/` and augments) and a `tts_edge.py`
(edge-tts wrapper), and modify `gen_data.py` to (1) also synthesize with edge-tts voices
and (2) fold in `real_data`. Keep `train.py`'s architecture/export contract; you may bump
channels and add dropout/SpecAugment.

**Approach B:** write one new `build_dataset.py` that produces `/tmp/ww/X_*.npy` /
`Y_*.npy` in the same format `train.py` already consumes. Either is fine.

Below are **reference snippets — adapt and TEST them** (you can run Python here; verify
shapes and that `feat.logmel` output is `(98,40)`).

### 5a. Neural TTS (edge-tts), English + Hebrew voices
```python
# tts_edge.py — natural voices; far closer to real speech than espeak.
import edge_tts, asyncio, subprocess, tempfile, os, numpy as np, soundfile as sf
EN = ["en-US-AriaNeural","en-US-GuyNeural","en-US-JennyNeural","en-US-AnaNeural",  # AnaNeural = child-ish
      "en-GB-RyanNeural","en-GB-SoniaNeural","en-AU-NatashaNeural","en-IN-NeerjaNeural"]
HE = ["he-IL-AvriNeural","he-IL-HilaNeural"]   # Hebrew-accented — matches the real kids
async def _save(text,voice,rate,pitch,path):
    await edge_tts.Communicate(text,voice,rate=rate,pitch=pitch).save(path)
def say_edge(text, voice, rate="+0%", pitch="+0Hz", out_sr=16000):
    mp3=tempfile.mktemp(suffix=".mp3"); wav=tempfile.mktemp(suffix=".wav")
    asyncio.run(_save(text,voice,rate,pitch,mp3))
    subprocess.run(["ffmpeg","-y","-i",mp3,"-ar",str(out_sr),"-ac","1",wav],
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
    a,_=sf.read(wav,dtype="float32"); os.remove(mp3); os.remove(wav)
    return a if a.ndim==1 else a[:,0]
# Positives: say_edge("hey pip", v, rate in {-15%..+20%}, pitch in {-10Hz..+30Hz}) over EN+HE.
#   Optionally also synthesize the Hebrew spelling "היי פיפ" with HE voices for accent coverage.
# Hard negatives (class 2): "hey","pip","hey kid","hey pick","pizza","hello","seven", kid answers, etc.
```

### 5b. Augmentation (numpy/scipy/librosa)
```python
# augment.py
import numpy as np
from scipy.signal import fftconvolve, butter, lfilter
import feat
SR, CLIP = feat.SR, feat.CLIP
def _rir():
    L=np.random.randint(200,1600); ir=np.exp(-np.arange(L)/np.random.uniform(40,300))
    ir*=(np.random.rand(L)<0.5); ir[0]=1.0; return ir.astype(np.float32)
def _bandlimit(x):  # mimic covered mic / tiny speaker
    lo=np.random.uniform(3000,7000); b,a=butter(4,lo/(SR/2),btype="low"); x=lfilter(b,a,x)
    if np.random.rand()<0.5: b,a=butter(2,120/(SR/2),btype="high"); x=lfilter(b,a,x)
    return x.astype(np.float32)
def _varispeed(u):  # changes speed+pitch together (cheap, no librosa)
    f=np.random.uniform(0.9,1.1); n=max(8,int(len(u)/f))
    return np.interp(np.linspace(0,len(u)-1,n),np.arange(len(u)),u).astype(np.float32)
def place_in_clip(u):  # random offset into a 1 s buffer
    u=u[:CLIP-100] if len(u)>CLIP-100 else u
    buf=np.zeros(CLIP,np.float32); off=np.random.randint(0,max(1,CLIP-len(u))); buf[off:off+len(u)]+=u
    return buf
def rms(x): return float(np.sqrt(np.mean(x*x))+1e-12)
def augment_speech(u, noise_pool):   # u: a clean utterance (float32). Returns a 1 s clip.
    s=u.astype(np.float32).copy()
    if np.random.rand()<0.5: s=_varispeed(s)
    if np.random.rand()<0.35: s=fftconvolve(s,_rir())[:len(s)+200]
    s=_bandlimit(s)
    s=s/(np.max(np.abs(s))+1e-9)*np.random.uniform(0.25,1.0)   # level — MATCHES device (§1.5)
    buf=place_in_clip(s)
    n=noise_pool[np.random.randint(len(noise_pool))]           # real OR synthetic 1 s noise
    snr=np.random.uniform(3,28); buf=buf+n*(rms(buf)/(10**(snr/20))/ (rms(n)+1e-9))
    m=np.max(np.abs(buf));  buf=buf/m if m>1 else buf
    return buf
```

### 5c. Real recordings → (clip, label)
```python
# real_data.py
import glob, numpy as np, soundfile as sf, feat, augment
def _load(p):
    a,sr=sf.read(p,dtype="float32"); a=a if a.ndim==1 else a[:,0]
    assert sr==16000, p
    return a
def windows_1s(a, hop=8000):  # slice long clips into overlapping 1 s windows
    out=[]; 
    for s in range(0,max(1,len(a)-feat.CLIP+1),hop): out.append(a[s:s+feat.CLIP])
    if len(a)<feat.CLIP: out.append(np.pad(a,(0,feat.CLIP-len(a))))
    return out
def collect(ww_dir, noise_pool, aug_per_pos=40):
    X=[];Y=[]
    for p in glob.glob(f"{ww_dir}/pos/p*.wav"):          # label 1, augment hard
        u=_load(p); u=u/(np.max(np.abs(u))+1e-9)
        for _ in range(aug_per_pos): X.append(feat.logmel(augment.augment_speech(u,noise_pool)));Y.append(1)
    for p in glob.glob(f"{ww_dir}/neg/n*.wav"):          # label 2 (other speech)
        u=_load(p); u=u/(np.max(np.abs(u))+1e-9)
        for _ in range(20): X.append(feat.logmel(augment.augment_speech(u,noise_pool)));Y.append(2)
    real_noise=[]
    for p in glob.glob(f"{ww_dir}/neg/c*.wav"):          # label 0 ambient → also feeds noise_pool
        for w in windows_1s(_load(p)):
            real_noise.append(w/(np.max(np.abs(w))+1e-9)*np.random.uniform(0.2,0.9))
            X.append(feat.logmel(w)); Y.append(0)
    return np.array(X,np.float32), np.array(Y,np.int64), real_noise
# Use the returned real_noise to extend noise_pool so SYNTHETIC speech is mixed with REAL room noise.
```

### 5d. Assemble & balance
- Build a `noise_pool` = synthetic 1 s noises (reuse `gen_data.rnoise`) **+** the real
  ambient windows from 5c. Mix it into every augmented speech clip.
- Generate counts (tune to your time budget): positives **real-aug ~5–8k** +
  **edge-TTS ~10–15k**; class-2 negatives **~15–20k** (TTS words + real `n` clips,
  heavily augmented + near-misses); class-0 noise **~8–12k** (real ambient windows +
  synthetic). Don't let synthetic drown the real positives — oversample/replicate real.
- Save chunks as `/tmp/ww/X_c0.npy`,`Y_c0.npy`, … so the existing `train.py`
  (`glob('/tmp/ww/X_c*.npy')`) picks them up unchanged. Also `np.savez('/tmp/ww/melfb.npz',
  fb=feat.MEL_FB,hann=feat.HANN)`.

---

## 6. Train
Use the existing `train.py` (keep its export contract). Recommended tweaks:
- Bump channels e.g. `Conv2D(24)→Conv2D(40)→Conv2D(64)` (still tiny; export handles it).
- Add `layers.Dropout(0.2)` before the dense, and a SpecAugment lambda on the spectrogram
  input during training (mask up to ~10 mel bins / ~15 frames).
- Keep `class_weight={0:1.,1:2.,2:1.}`, ~45 epochs, ReduceLROnPlateau.
- It already prints **val accuracy, a confusion matrix, and a threshold sweep**
  (precision/recall/false-fires at thr 0.5/0.7/0.8/0.9). **Record the threshold sweep** —
  you'll recommend a `WAKE_WORD_THRESHOLD` from it.
```bash
cd ESP32/wake_word_training && source ~/wwenv/bin/activate
python3 build_dataset.py        # your 5a–5d (or: python3 gen_data.py <args> + your additions)
python3 train.py                # → /tmp/ww/model.keras + /tmp/ww/norm.npz
```
**Target:** val acc ≥ ~0.97 AND, in the threshold sweep, **high hey-pip recall with ~0
false-fires at thr≈0.7–0.8**. If false-fires are high, add more/real negatives (don't just
raise the threshold).

---

## 7. Export + PARITY GATE (must pass before shipping)
```bash
python3 export_c.py             # writes /tmp/ww/out/hey_pip_model.h
# parity.py imports tts.py (espeak). With espeak-ng installed (Section 3) it runs as-is.
cp ww_infer.h host_test.c /tmp/ww/out/ 2>/dev/null; cp ww_infer.h host_test.c /tmp/ww/ 2>/dev/null
cd /tmp/ww && cc -O2 -o out/host_test host_test.c -I/tmp/ww/out -lm   # build C harness against the NEW header
cd ESP32/wake_word_training && python3 parity.py   # MUST print: PARITY: PASS (<1e-3)
```
If parity is not PASS, **do not ship** — the C front-end/weights diverge from Keras.
Debug (usually a layout/normalization mismatch in your export or a `feat.py` edit you
shouldn't have made). Note: `host_test.c` hardcodes scratch dims `b1[48*19*..]`,
`b2[23*9*..]`, `b3[11*4*..]` for the 98×40 front-end — those stay valid as long as you
kept `feat.py` unchanged (§1.1).

---

## 8. Deploy
```bash
cp /tmp/ww/out/hey_pip_model.h  ESP32/homework_assistant/hey_pip_model.h
```
Then tell the user (they flash on Windows):
1. Reflash the firmware with the new `hey_pip_model.h` (and `WW_CAPTURE_MODE 0`).
2. **Re-tune the threshold on the device:** build with `WW_TEST_MODE 1`
   (`homework_assistant.ino`), say "hey pip", read the `max` score column, set
   `WAKE_WORD_THRESHOLD` (in `wake_word.h`) just below it — start from the value your
   §6 threshold sweep recommended. The front-end gates/gains there were already tuned
   (`WW_MIN_PEAK 0.015`, `WW_MAX_GAIN 50`, `WW_CONSEC 2`); a better model may let you
   raise the threshold back toward 0.8 for fewer false fires.
3. Report back: detection rate on real "hey pip", and whether the bed-noise false fire
   is gone. Commit the new header + your training scripts.

---

## 9. If it's still not robust — levers (in priority order)
1. **More/again real recordings** — especially the specific noises that false-fire, and
   the real users (kids). This beats everything else.
2. **More negative diversity** (class 0 & 2) — false fires are almost always a weak
   negative set, not a too-low threshold.
3. **Heavier/realistic augmentation** — make sure the band-limit/EQ + real-noise mixing
   actually matches the device's covered mic (compare a real `ww/pos` log-mel to a
   synthetic one; close the gap).
4. **A bit more model capacity** (channels) — only after data is good; watch it still
   fits PSRAM/CPU on-device (it's tiny, headroom is large).
5. Re-tune device front-end (`wake_word.h`) as a last step, not a first.

---

## Appendix — file/contract quick reference
- `feat.py` — log-mel front-end. **MUST match `ww_infer.h`. Don't change.**
- `tts.py` — espeak (legacy). Keep for `parity.py`; add `tts_edge.py` for real training.
- `gen_data.py` — synthetic data gen. Extend (edge-TTS + real data) or replace with `build_dataset.py`.
- `train.py` — trains the CNN, writes `/tmp/ww/model.keras` + `norm.npz`, prints metrics. Channels OK to bump.
- `export_c.py` — writes `hey_pip_model.h` (weights+melfb+MEAN/STD+dims). **Keep format.**
- `parity.py` + `host_test.c` + `ww_infer.h` — C-vs-Keras parity gate. Must PASS.
- Device side (don't edit unless asked): `../homework_assistant/hey_pip_model.h` (output target),
  `../homework_assistant/wake_word.h` (runtime peak-norm to ~0.60 + thresholds),
  `../homework_assistant/ww_capture.h` (how the user recorded `ww/`).
- Class order (fixed): **0=noise, 1=hey-pip, 2=other-speech.** Firmware reads `prob[1]`.

---

## Addendum (2026-07-03): level-invariant front-end (CMN) + device fixes

### Training pipeline changes

- **`feat.py` logmel now applies per-band cepstral mean normalization (CMN).**
  After computing the 98×40 log-mel, each of the 40 mel bands has its mean over
  the 98 frames subtracted, removing any stationary level offset or spectral tilt
  before the CNN sees the data. `ww_infer.h` mirrors this step exactly, gated by
  `#define WW_CMN 1`; the define is emitted by `export_c.py` into the generated
  `hey_pip_model.h`. **The parity gate (`parity.py`) verifies the C and Keras
  front-ends agree through CMN** — it must PASS before shipping. The `/tmp/ww`
  workspace and parity build line are unchanged:
  ```bash
  cc -O2 -o out/host_test host_test.c -I/tmp/ww/out -lm
  ```

- **`augment.py` level range widened to 0.10–1.0** (was 0.25–1.0). CMN removes the
  absolute level anyway, so training with a wider range improves robustness rather
  than confusing the model.

- **`eval_device.py` (new) replaces ad-hoc threshold evaluation.** Run it after
  `train.py` (before `export_c.py` is strictly necessary, but after the model is
  trained). It simulates the exact firmware streaming pipeline — rolling 1 s window,
  100 ms hop, peak-norm, noise gate (`WW_MIN_PEAK`), 2-consecutive-poll debounce —
  over the real `ww/` recordings, and prints recall / false-fire tables per threshold.
  Pick `WAKE_WORD_THRESHOLD` from its output and record it in `wake_word.h`.

### Device-side changes (already in firmware — no model retraining needed for these)

- **ES8311 REG16 listen boost (+18 dB).** While wake-listening, the firmware raises
  `ADC_SCALE[2:0]` in REG16 from 24 dB to 42 dB (`es8311SetWakeListenBoost()` in
  `es8311.h`). This happens inside the codec before 16-bit I2S truncation, recovering
  ~3 bits of resolution from quiet/distant speech. It is restored to 24 dB the moment
  listening stops, so answer-recording and STT capture levels are completely unaffected.

- **Per-poll dual-slot speech-likeness mic picking.** Instead of pinning a fixed I2S
  slot, `wake_word.h` keeps both stereo slots' rolling 1 s windows. Each poll selects
  the slot whose coefficient of variation of 50 ms frame RMS is higher (speech
  modulates; the crosstalk slot is near-stationary), with fallback to the slot
  confirmed by the last STT capture (`wakeWordNoteSttChannel`), then to
  `WW_MIC_CHANNEL`. A wrong pick can only occur on windows with no speech, which the
  noise gate suppresses anyway.

### Net effect on retrained models

Because CMN removes absolute level and the codec boost makes quiet speech visible,
**retrained models no longer require device-side level re-tuning** after deployment.
The workflow after retraining is: run `eval_device.py`, copy the recommended threshold
into `WAKE_WORD_THRESHOLD` in `wake_word.h`, reflash, and verify with `WW_TEST_MODE 1`.
