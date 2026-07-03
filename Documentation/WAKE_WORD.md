# Wake Word — "hey pip" (self-contained, no library)

Replaces the push-to-talk **button** with an on-device **wake word**: the device
listens continuously on the ES8311 mic, and when the child says **"hey pip"** it
records the answer (ended automatically on silence) instead of waiting for a
button hold.

The model was **trained offline for you** and is bundled in the sketch — there is
**nothing to install** (no Edge Impulse, no TensorFlow, no Arduino ML library).
The button is **not deleted**: it is compiled out behind `USE_WAKE_WORD` and comes
back by flipping one flag (see [Reverting](#reverting-to-the-button)).

---

## Files

| File | What it is |
|------|------------|
| `hey_pip_model.h` | Auto-generated weights: 40-bin mel filterbank + the CNN (~21k floats, ~83 KB flash). Also contains `#define WW_CMN 1`, enabling per-band cepstral mean normalization in the front-end. |
| `ww_infer.h` | Pure-C inference: log-mel front-end (FFT + mel + log + CMN) and the CNN forward pass. No dependencies. |
| `wake_word.h` | Arduino glue: streams the ES8311 mic into rolling 1 s windows, manages codec listen-boost and dual-slot mic picking, runs inference. Exposes `wakeWordBegin/StartListening/Poll/StopListening`. |
| `ESP32/wake_word_training/` | The full training pipeline used to make the model, so you can retrain/improve it later. |

---

## How it works

1. While listening, the firmware raises the ES8311's ADC scale-up register (REG16,
   `ADC_SCALE[2:0]`) from its 24 dB default to 42 dB (`es8311SetWakeListenBoost()`).
   This +18 dB happens inside the codec before 16-bit I2S truncation, recovering
   roughly 3 bits of resolution from quiet or distant speech (raw peak ~0.01 → ~0.10).
   The boost is restored to 24 dB the moment listening stops, so recording/STT levels
   are completely unaffected.
2. Both stereo I2S slots are kept as rolling 1-second windows. Each poll picks the
   slot whose 50 ms frame-RMS **coefficient of variation** is higher (speech
   modulates; the other slot carries only steady crosstalk), falling back to the
   slot confirmed by the last successful STT capture, then to `WW_MIC_CHANNEL`.
3. The chosen window is peak-normalised toward `WW_TARGET_PEAK`, then a
   **log-mel spectrogram** (98 × 40) is computed. **Per-band cepstral mean
   normalization (CMN)** subtracts each mel band's mean over the 98 frames, removing
   any overall level offset and stationary spectral tilt (covered mic, codec gain
   drift, etc.). The CNN scores the *shape* of the sound, not its loudness.
4. A **3-class CNN** (channels 32/48/64, GlobalAveragePooling, Dense(3)) outputs
   probabilities: `0 = noise`, `1 = hey pip`, `2 = other speech`.
5. If `P(hey pip) ≥ WAKE_WORD_THRESHOLD` on `WW_CONSEC` consecutive polls the word
   fires, I2S hands off to the recorder, and the existing answer→STT→tutor pipeline
   runs unchanged.

I2S is installed only while listening and released the instant the wake word fires,
so it never clashes with the recorder or the TTS player (which install their own
I2S). The wake-word path reuses `i2s_start_recording()`, so the ES8311 clocking you
tuned is identical for listening and recording.

---

## Build & flash (nothing to install)

1. Make sure `USE_WAKE_WORD` is `1` in `homework_assistant.ino` (it is by default).
2. Arduino IDE board settings (same as before):
   - Board **ESP32S3 Dev Module**, Flash **16MB**, **PSRAM: OPI PSRAM** (required —
     both slot windows and model scratch live in PSRAM), Partition **Huge APP**.
3. Upload. On boot the serial log prints:
   ```
   [WakeWord] 'hey pip' model ready: 98-frame log-mel, CNN 32/48/64, CMN on
   ```
4. Say **"hey pip"**, then answer.

---

## Tuning (in `wake_word.h`)

| Macro | Default | Effect |
|-------|---------|--------|
| `WAKE_WORD_THRESHOLD` | `0.90` (from `eval_device.py`, 2026-07-03: recall 149/149 usable positives, 0 ambient false fires in 10 min, 5/200 other-speech files) | Confidence to fire. Raise toward 0.95 to cut false fires; lower toward 0.80 to catch more. |
| `WW_TARGET_PEAK` | `0.60` | Target peak for window normalisation before the CNN. |
| `WW_MAX_GAIN` | `25` | Cap on the normalisation gain multiplier. |
| `WW_MIN_PEAK` | `0.040` | Raw peak floor on the boosted signal; windows below this are noise-gated. Boosted quiet-room floor ≈ 0.03; boosted distant speech ≈ 0.10. Lower if `[gated]` appears while speaking. |
| `WW_CONSEC` | `3` | Consecutive polls above threshold to fire (300 ms debounce). Halved other-speech false fires vs 2 at zero recall cost. |
| `WW_HOP` | `1600` (100 ms) | Audio samples pulled per poll = detection granularity vs. CPU. |

### Reading `WW_TEST_MODE 1` output

Build with `#define WW_TEST_MODE 1` in `homework_assistant.ino` to get one line per
~100 ms on the serial port:

```
p0=0.012 p1=0.103  cv0=0.41 cv1=1.83  ch=1  win=0.103>0.040  gain=5.8  score=0.87  max=0.91  FIRE
```

- **p0 / p1** — per-hop peak of each stereo slot. The slot whose `p` jumps when you
  talk is the mic.
- **cv0 / cv1** — speech-likeness of each slot (coefficient of variation of 50 ms
  frame RMS). The mic slot will have a higher CV while you speak.
- **ch** — which slot was scored this poll.
- **win** — chosen window raw peak vs `WW_MIN_PEAK`. If this shows `[gated]` while
  you are speaking, lower `WW_MIN_PEAK`.
- **gain** — normalisation multiplier applied to the window.
- **score** — `P(hey pip)` from the CNN for this poll.
- **max** — highest score seen since the last reset.
- Tags: `[gated]` = noise-gated; `(arming)` = one poll above threshold;
  `FIRE` = fired.

After a few "hey pip" utterances, note `max` and set `WAKE_WORD_THRESHOLD` just
below it.

---

## Accuracy & honest caveats

- **C/Python parity:** the on-device C inference matches the Python trainer
  **bit-for-bit (max diff 1e-6)** on held-out clips. This is enforced by `parity.py`
  before every model ship.
- The model (June 2026, front-end refreshed July 2026) is anchored on **real "hey
  pip" recordings** made through this device's own mic (`ww/pos/`), real other-speech,
  and real ambient captures — the old "trained only on synthetic voices" caveat is
  obsolete. Synthetic neural-TTS voices (English + Hebrew-accented) and espeak hard
  negatives are used for diversity, not as the primary data.
- **Eval tables are printed by `eval_device.py`**, not hard-coded here. Run it after
  retraining to get recall / false-fire rates per threshold, then pick
  `WAKE_WORD_THRESHOLD` from its output.
- Because CMN removes absolute level, the model is robust to mic-distance variation
  and codec-gain drift without re-tuning the threshold.

---

## Footprint

- Flash: ~83 KB of weights (mostly the mel filterbank).
- PSRAM scratch: ~230 KB total (two 1 s slot windows + activations), allocated once
  at boot (~+64 KB vs. a single-slot design for the second window).
- Compute: ~30–40 ms per inference, run every ~100 ms while idle (a few % CPU).

---

## Retrain / improve the model

Everything used to build the model is in `ESP32/wake_word_training/`. The pipeline
requires a **conda/miniforge environment** (Python 3.11 recommended) with:
`tensorflow-cpu numpy scipy soundfile librosa edge-tts espeakng_loader py-espeak-ng`.
Note that `edge-tts` needs internet access (Microsoft neural voices).

To regenerate `hey_pip_model.h`:

```bash
# Activate your miniforge env first, e.g.:  conda activate wwenv
python3 build_dataset.py   # assembles real recordings + edge-TTS + noise + augmentation
python3 train.py           # trains CNN 32/48/64, saves model.keras + norm.npz
python3 export_c.py        # writes hey_pip_model.h (includes WW_CMN define)
python3 parity.py          # C-vs-Keras gate — MUST print PARITY: PASS (<1e-3)
python3 eval_device.py     # simulates firmware streaming → prints recall/false-fire tables per threshold
```

`eval_device.py` simulates the exact firmware pipeline (rolling window, 100 ms hop,
peak-norm, noise gate, 2-consecutive debounce) over the real `ww/` recordings and
prints per-threshold recall / false-fire tables. **Pick `WAKE_WORD_THRESHOLD` from
its output**, not from training metrics.

---

## Reverting to the button

Set one flag in `homework_assistant.ino` and re-flash:

```cpp
#define USE_WAKE_WORD 0
```

That recompiles the original push-to-talk path (button GPIO + button-held capture)
and ignores the wake-word model entirely.
