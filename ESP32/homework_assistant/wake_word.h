#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Wake-word module — English "hey pip" (self-contained, no external library)
//
// Replaces the push-to-talk button: the device listens continuously on the
// ES8311 mic and fires when it hears "hey pip"; the main loop then records the
// child's answer (ended automatically by trailing silence).
//
// HOW IT WORKS — fully on-device, NOTHING to install:
//   • A small CNN (retrained on the REAL /ww recordings from this device's mic,
//     plus neural-TTS voices and hard negatives) runs on a log-mel spectrogram
//     of the last 1 second of audio. Both the front-end and the network are
//     plain C in ww_infer.h + hey_pip_model.h — no TensorFlow, no Edge Impulse.
//     The exact same math is verified bit-for-bit against the trainer on a host
//     PC (parity.py).
//   • The features are level-invariant (per-band CMN, see ww_infer.h): the model
//     scores the SHAPE of the sound, not its absolute loudness, so mic-level
//     drift (covered mic hole, codec gain changes) no longer silently kills it.
//   • Streaming: each poll reads ~100 ms of new mic audio into rolling 1 s
//     windows and runs one inference; prob["hey pip"] ≥ WAKE_WORD_THRESHOLD on
//     WW_CONSEC consecutive polls fires.
//
// THE TWO DEVICE-SIDE FIXES THAT MADE IT WORK (July 2026):
//   1. CODEC LISTEN GAIN — the ES8311's ADC scale-up (REG16) idled at 24dB while
//      the analog PGA was already maxed; a distant "hey pip" reached the CPU at
//      peak ~0.01, i.e. ~8 bits of real signal. While listening we now raise
//      REG16 to its 42dB max (es8311SetWakeListenBoost), +18dB of REAL resolution,
//      and restore it before the answer is recorded so STT levels are untouched.
//   2. NO MORE PINNED/GUESSED MIC SLOT — the ES8311 is mono inside a stereo I2S
//      frame; the other slot carries only steady clock/common-mode crosstalk
//      (crest ≈1, no modulation). Instead of pinning one slot (deaf forever if
//      wrong) or energy-picking at boot (crosstalk out-energizes a quiet room),
//      we keep BOTH slots' rolling windows and pick per poll by SPEECH-LIKENESS
//      (variance of frame RMS — speech modulates, crosstalk doesn't), falling
//      back to the slot the last successful STT capture proved is the mic
//      (wakeWordNoteSttChannel). A wrong pick is only possible when the window
//      holds no speech at all — exactly when the noise gate squelches anyway.
//
// TUNING: flash with WW_TEST_MODE 1, watch both slots' levels + the live score,
// then adjust WAKE_WORD_THRESHOLD / WW_MIN_PEAK. See Documentation/WAKE_WORD.md.
// ─────────────────────────────────────────────────────────────────────────────

#include <Arduino.h>
#include <driver/i2s.h>
#include <string.h>
#include "pins.h"
#include "ww_infer.h"     // pure-C log-mel(+CMN) + CNN (includes hey_pip_model.h)

// Confidence (0..1) on the "hey pip" class needed to trigger. Set from the host
// eval on the real /ww recordings (wake_word_training/eval_device.py, 2026-07-03):
// at 0.90 with WW_CONSEC=3, real-"hey pip" recall was 149/149 usable files
// (5th-percentile debounced score 0.993 — huge margin), 0 false fires over 10 min
// of real ambient, 5/200 other-speech files. Raise toward 0.95 if it false-fires
// on speech; lower toward 0.80 if it misses (watch 'score'/'max' in WW_TEST_MODE).
#ifndef WAKE_WORD_THRESHOLD
#define WAKE_WORD_THRESHOLD 0.90f
#endif

// ── Window leveling ───────────────────────────────────────────────────────────
// The 1 s window is peak-normalized to WW_TARGET_PEAK before inference. With the
// CMN front-end the model no longer NEEDS an exact level — this only keeps the
// log-mel epsilon floor in the same regime as training (clips at peak 0.10–1.0).
#ifndef WW_TARGET_PEAK
#define WW_TARGET_PEAK 0.60f
#endif
// Gain cap so a silent room's floor is never blown up to speech level. With the
// +18dB codec listen boost, a distant "hey pip" reaches peak ~0.08–0.4 → typical
// gain 1.5–8×; the cap only engages on near-silence (which the gate rejects).
#ifndef WW_MAX_GAIN
#define WW_MAX_GAIN 25.0f
#endif

// ── False-fire suppression ────────────────────────────────────────────────────
// NOISE GATE: ignore any window whose RAW peak (on the chosen mic slot, with the
// listen boost active) is below this. Boosted quiet-room floor measured ~0.03,
// boosted distant speech ~0.10 — the gate sits between. Raise it if a quiet room
// false-fires; lower it if a soft wake word is ignored (watch 'win' in test mode).
#ifndef WW_MIN_PEAK
#define WW_MIN_PEAK 0.040f
#endif
// DEBOUNCE: a real wake word stays in the 1 s window for several polls, so it
// scores high on WW_CONSEC polls in a row; stray noise blips don't. 3 polls =
// 300 ms of sustained confidence: in the host eval this halved other-speech
// false fires vs 2, at ZERO recall cost (real positives hold ≥0.99 for many
// polls). Only lower to 2 if very fast/clipped "hey pip" utterances get missed.
#ifndef WW_CONSEC
#define WW_CONSEC 3
#endif

// New audio pulled in per poll (samples @16 kHz). 1600 = 100 ms → ~100 ms detect
// granularity; the model still sees the full 1 s window each time.
#ifndef WW_HOP
#define WW_HOP 1600
#endif

// Fallback stereo slot for the mic when neither the speech-likeness picker nor a
// past STT capture can decide (i.e. the window is essentially silent — harmless,
// the gate squelches those). On this board the mic was observed on slot 1.
#ifndef WW_MIC_CHANNEL
#define WW_MIC_CHANNEL 1
#endif

// ── Reused from the main sketch (identical ES8311 I2S clocking for listen+record)
void i2s_start_recording();
void i2s_stop_recording();
void faceTick();
// es8311SetWakeListenBoost() comes from es8311.h, included before us in the .ino.

// ── State / PSRAM scratch ─────────────────────────────────────────────────────
static bool   ww_active = false;
static int    ww_stt_ch = -1;      // mic slot proven by the last STT capture (-1 = none yet)
static int    ww_last_ch = WW_MIC_CHANNEL;   // slot used for the most recent inference
static int    ww_consec = 0;       // consecutive over-threshold polls (debounce)
static float* ww_win[2] = {nullptr, nullptr}; // rolling 1 s window per stereo slot
static float* ww_sig  = nullptr;   // gain-normalized copy fed to inference (WW_CLIP)
static float* ww_feat = nullptr;   // log-mel             (WW_FRAMES*WW_MELS)
static float* ww_b1 = nullptr;     // conv1 activations
static float* ww_b2 = nullptr;     // conv2 activations
static float* ww_b3 = nullptr;     // conv3 activations

static inline int ww_oh(int ih){ return (ih-WW_K)/2+1; }

// Allocate scratch (PSRAM) and print model info. Call once at boot.
inline bool wakeWordBegin(){
  int H1=ww_oh(WW_FRAMES), W1=ww_oh(WW_MELS);
  int H2=ww_oh(H1),        W2=ww_oh(W1);
  int H3=ww_oh(H2),        W3=ww_oh(W2);
  ww_win[0] = (float*)ps_malloc(sizeof(float)*WW_CLIP);
  ww_win[1] = (float*)ps_malloc(sizeof(float)*WW_CLIP);
  ww_sig  = (float*)ps_malloc(sizeof(float)*WW_CLIP);
  ww_feat = (float*)ps_malloc(sizeof(float)*WW_FRAMES*WW_MELS);
  ww_b1   = (float*)ps_malloc(sizeof(float)*H1*W1*WW_C1_OUT);
  ww_b2   = (float*)ps_malloc(sizeof(float)*H2*W2*WW_C2_OUT);
  ww_b3   = (float*)ps_malloc(sizeof(float)*H3*W3*WW_C3_OUT);
  if(!ww_win[0]||!ww_win[1]||!ww_sig||!ww_feat||!ww_b1||!ww_b2||!ww_b3){
    Serial.println("[WakeWord] ❌ PSRAM alloc failed.");
    return false;
  }
  Serial.printf("[WakeWord] 'hey pip' model ready: %d-frame log-mel%s, CNN %d/%d/%d, threshold %.2f\n",
                WW_FRAMES,
#ifdef WW_CMN
                "+CMN",
#else
                "",
#endif
                WW_C1_OUT, WW_C2_OUT, WW_C3_OUT, (double)WAKE_WORD_THRESHOLD);
  return true;
}

// The .ino calls this after every successful answer capture with the stereo slot
// its (loud-speech, reliable) energy pick chose — ground truth for our picker.
inline void wakeWordNoteSttChannel(int ch){
  if(ch==0 || ch==1) ww_stt_ch = ch;
}

// Begin listening: raise the codec listen gain, install I2S, flush the first DMA
// buffer (codec-settling garbage), clear the rolling windows.
inline void wakeWordStartListening(){
  if(ww_active) return;
  es8311SetWakeListenBoost(true);          // +18dB ADC scale-up while we listen
  i2s_start_recording();
  { const int FR=1024; static int16_t tmp[FR*2]; size_t br=0;
    i2s_read(I2S_PORT,(char*)tmp,sizeof(tmp),&br,portMAX_DELAY); }
  memset(ww_win[0],0,sizeof(float)*WW_CLIP);
  memset(ww_win[1],0,sizeof(float)*WW_CLIP);
  ww_consec = 0;
  ww_active = true;
}

// Stop listening and release I2S for the recorder / TTS player. Restores the
// codec to the exact STT levels BEFORE the recorder reinstalls I2S.
inline void wakeWordStopListening(){
  if(!ww_active) return;
  i2s_stop_recording();
  es8311SetWakeListenBoost(false);
  ww_active=false;
}

// Slide both rolling windows by WW_HOP and read fresh stereo samples into their
// tails. Returns via out-params the per-hop RMS/peak of each slot (test mode).
static void ww_read_hop(float* rms0, float* pk0, float* rms1, float* pk1){
  memmove(ww_win[0], ww_win[0]+WW_HOP, (WW_CLIP-WW_HOP)*sizeof(float));
  memmove(ww_win[1], ww_win[1]+WW_HOP, (WW_CLIP-WW_HOP)*sizeof(float));
  float* t0 = ww_win[0]+(WW_CLIP-WW_HOP);
  float* t1 = ww_win[1]+(WW_CLIP-WW_HOP);
  float ss0=0, ss1=0, p0=0, p1=0;
  int got=0; int16_t st[256*2];
  while(got<WW_HOP){
    int want=WW_HOP-got, frames=want<256?want:256; size_t br=0;
    i2s_read(I2S_PORT,(char*)st,frames*4,&br,portMAX_DELAY);
    int f=(int)(br/4);
    for(int i=0;i<f && got<WW_HOP;i++){
      float a = st[i*2+0]/32768.0f, b = st[i*2+1]/32768.0f;
      t0[got]=a; t1[got]=b; got++;
      ss0+=a*a; ss1+=b*b;
      if(fabsf(a)>p0) p0=fabsf(a);
      if(fabsf(b)>p1) p1=fabsf(b);
    }
    faceTick();
  }
  if(rms0) *rms0 = sqrtf(ss0/(float)WW_HOP);
  if(pk0)  *pk0  = p0;
  if(rms1) *rms1 = sqrtf(ss1/(float)WW_HOP);
  if(pk1)  *pk1  = p1;
}

// SPEECH-LIKENESS of a 1 s window: coefficient of variation of its 50 ms frame
// RMS. Speech modulates strongly (CV >> 0); the crosstalk slot is a steady tone
// (CV ≈ 0). This is what lets us find the mic without trusting wiring lore.
static float ww_speechiness(const float* w){
  const int NF=20, FL=WW_CLIP/NF;
  float m=0.0f, m2=0.0f;
  for(int f=0;f<NF;f++){
    const float* p=w+f*FL; float acc=0.0f;
    for(int i=0;i<FL;i++) acc+=p[i]*p[i];
    float r=sqrtf(acc/(float)FL);
    m+=r; m2+=r*r;
  }
  m/=NF; m2/=NF;
  float var=m2-m*m; if(var<0) var=0;
  return sqrtf(var)/(m+1e-9f);
}

// Choose the slot to score this poll. Clear dynamics winner → take it; otherwise
// trust the slot the last STT capture proved; otherwise the wiring default.
static int ww_pick_channel(float cv0, float cv1){
  float hi = cv0>cv1?cv0:cv1, lo = cv0<cv1?cv0:cv1;
  if(hi > 1.3f*lo + 0.02f) return (cv1>cv0)?1:0;
  if(ww_stt_ch >= 0) return ww_stt_ch;
  return WW_MIC_CHANNEL;
}

// Scale a 1 s window to WW_TARGET_PEAK, capped at WW_MAX_GAIN. Writes to `out`
// (never touches the rolling window, so gain can't compound across polls).
static float ww_normalize(const float* in, float* out, float* gainOut){
  float peak = 1e-6f;
  for(int i=0;i<WW_CLIP;i++){ float a = fabsf(in[i]); if(a>peak) peak=a; }
  float g = WW_TARGET_PEAK / peak;
  if(g > WW_MAX_GAIN) g = WW_MAX_GAIN;
  if(g < 1.0f)        g = 1.0f;     // already loud enough — never attenuate
  for(int i=0;i<WW_CLIP;i++) out[i] = in[i]*g;
  if(gainOut) *gainOut = g;
  return peak;                      // RAW window peak (pre-gain) — noise-gate input
}

// Pull ~100 ms of new audio, slide the windows, run one inference on the mic slot.
// Returns 1 if "hey pip" detected this poll, 0 if not, -1 if not listening.
inline int wakeWordPoll(){
  if(!ww_active) return -1;
  ww_read_hop(nullptr,nullptr,nullptr,nullptr);

  float cv0 = ww_speechiness(ww_win[0]);
  float cv1 = ww_speechiness(ww_win[1]);
  int   ch  = ww_pick_channel(cv0, cv1);
  ww_last_ch = ch;

  float winpeak = ww_normalize(ww_win[ch], ww_sig, nullptr);
  if(winpeak < WW_MIN_PEAK){ ww_consec = 0; return 0; }     // room too quiet to be speech
  float prob[WW_NCLASS];
  ww_infer(ww_sig, ww_feat, ww_b1, ww_b2, ww_b3, prob);     // prob[1] = "hey pip"
  if(prob[1] >= WAKE_WORD_THRESHOLD){
    if(++ww_consec >= WW_CONSEC){                            // sustained → real
      ww_consec = 0;
      Serial.printf("[WakeWord] 'hey pip' %.2f (ch%d) ✓\n", (double)prob[1], ch);
      return 1;
    }
  } else {
    ww_consec = 0;
  }
  return 0;
}

// ── On-device TEST MODE — live score printer for tuning / debugging ───────────
// Never returns. One line per ~100 ms of audio with everything needed to diagnose
// an unresponsive wake word WITHOUT any cloud setup:
//   p0/p1   = per-hop peak of stereo slot 0 / slot 1 (0..1). The one that jumps
//             when you talk is the mic; the one stuck at a constant small value
//             is the crosstalk slot. BOTH ~0 while speaking ⇒ no audio at all.
//   cv0/cv1 = speech-likeness of each slot's 1 s window (modulation). The picker
//             takes the clearly higher one.
//   ch      = slot actually scored this poll.
//   win     = chosen window's RAW peak (pre-gain) — compare against WW_MIN_PEAK.
//   gain    = auto-level applied before inference (caps at WW_MAX_GAIN).
//   score   = model confidence for "hey pip"; max = highest so far this run.
// Tuning recipe: say "hey pip" a few times, note 'max', set WAKE_WORD_THRESHOLD
// a touch below it. If 'win' shows [gated] while you speak, lower WW_MIN_PEAK.
inline void wakeWordRunTestMode(){
  Serial.println("\n========== WAKE-WORD TEST MODE ==========");
  Serial.printf ("Threshold=%.2f  gate=%.3f  consec=%d  (codec listen boost ACTIVE: +18dB)\n",
                 (double)WAKE_WORD_THRESHOLD, (double)WW_MIN_PEAK, WW_CONSEC);
  Serial.println("Say \"hey pip\" and watch 'score'/'max'. This loop never exits;");
  Serial.println("reflash with WW_TEST_MODE 0 for normal use.\n");

  if(!ww_win[0]){
    Serial.println("[WWTEST] scratch not allocated — wakeWordBegin() failed (PSRAM?). Halting.");
    while(true){ faceTick(); delay(200); }
  }

  wakeWordStartListening();
  float maxScore = 0.0f;
  int   tmConsec = 0;
  while(true){
    float r0,p0,r1,p1;
    ww_read_hop(&r0,&p0,&r1,&p1);
    float cv0 = ww_speechiness(ww_win[0]);
    float cv1 = ww_speechiness(ww_win[1]);
    int   ch  = ww_pick_channel(cv0, cv1);
    float gain; float winpeak = ww_normalize(ww_win[ch], ww_sig, &gain);
    float prob[WW_NCLASS];
    ww_infer(ww_sig, ww_feat, ww_b1, ww_b2, ww_b3, prob);
    float score = prob[1];
    if(score > maxScore) maxScore = score;
    // Mirror the live gate + debounce so the display matches real behavior.
    const char* tag = "";
    if(winpeak < WW_MIN_PEAK){ tag = "  [gated: too quiet]"; tmConsec = 0; }
    else if(score >= WAKE_WORD_THRESHOLD){
      if(++tmConsec >= WW_CONSEC) tag = "  <<< FIRE";
      else                        tag = "  (arming)";
    } else { tmConsec = 0; }
    Serial.printf("[WWTEST] p0=%.3f p1=%.3f cv0=%.2f cv1=%.2f ch=%d win=%.3f gain=%4.1fx score=%.3f max=%.3f%s\n",
                  (double)p0,(double)p1,(double)cv0,(double)cv1,ch,
                  (double)winpeak,(double)gain,(double)score,(double)maxScore,tag);
  }
}
