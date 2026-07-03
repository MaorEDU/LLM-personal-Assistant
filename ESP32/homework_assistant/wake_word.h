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
//     of the last 1 second of audio (ww_infer.h + hey_pip_model.h, plain C,
//     verified bit-for-bit against the trainer by parity.py).
//   • The features are level-invariant (per-band CMN, see ww_infer.h): the model
//     scores the SHAPE of the sound, not its absolute loudness — so there is NO
//     per-window gain stage here at all anymore.
//   • Streaming, HARD-REAL-TIME: each poll ingests WW_HOP samples and computes
//     ONLY the log-mel frames that hop added (WW_HOP/WW_FRAME_STEP rows) into a
//     rolling feature buffer — not all 98 frames. This matters: the full
//     recompute + this CNN took ~150 ms per poll on the ESP32-S3, which OVERRAN
//     the 100 ms hop, overflowed the I2S DMA, and silently dropped audio — the
//     model heard "hey pip" time-compressed ~3× and never recognized it. That
//     was the final root cause of "no response" after all the gain fixes.
//     Budget now: ~20 incremental FFTs + 1 CNN ≈ 70–100 ms per 200 ms hop.
//
// THE OTHER TWO DEVICE-SIDE FIXES (July 2026):
//   1. CODEC LISTEN GAIN — ES8311 REG16 ADC scale-up raised 24→42 dB while
//      listening (es8311SetWakeListenBoost), restored before recording so the
//      STT path is untouched. +18 dB of REAL resolution before 16-bit I2S.
//   2. NO PINNED MIC SLOT — both stereo slots keep rolling windows; each poll
//      picks by speech-likeness (variance of frame RMS), seeded by the slot the
//      last successful STT capture proved (wakeWordNoteSttChannel). On current
//      boards both slots carry identical mono ADC data, so the pick is moot —
//      but it stays robust to board/driver revisions where they differ.
//
// TUNING: flash with WW_TEST_MODE 1, watch both slots' levels + the live score,
// then adjust WAKE_WORD_THRESHOLD / WW_MIN_PEAK. The 'dt' column MUST stay at
// ~WW_HOP/16 ms (200 ms default): if dt runs consistently higher, the loop is
// overrunning and audio is being dropped — raise WW_HOP. See WAKE_WORD.md.
// ─────────────────────────────────────────────────────────────────────────────

#include <Arduino.h>
#include <driver/i2s.h>
#include <string.h>
#include "pins.h"
#include "ww_infer.h"     // pure-C log-mel(+CMN) + CNN (includes hey_pip_model.h)

// Confidence (0..1) on the "hey pip" class needed to trigger. Set from the host
// eval on the real /ww recordings (wake_word_training/eval_device.py) at the
// current WW_HOP/WW_CONSEC. Raise toward 0.95 if it false-fires on speech;
// lower toward 0.80 if it misses (watch 'score'/'max' in WW_TEST_MODE).
#ifndef WAKE_WORD_THRESHOLD
#define WAKE_WORD_THRESHOLD 0.90f
#endif

// ── False-fire suppression ────────────────────────────────────────────────────
// NOISE GATE: skip scoring while the 1 s window's RAW peak (chosen slot, listen
// boost active) is below this. Boosted quiet-room floor measured ~0.02–0.03 on
// device; a boosted real "hey pip" ~0.33–0.44. Raise if a quiet room false-
// fires; lower if a soft wake word shows [gated] in test mode while speaking.
#ifndef WW_MIN_PEAK
#define WW_MIN_PEAK 0.040f
#endif
// DEBOUNCE: consecutive over-threshold polls to fire. 2 polls @200 ms hop =
// 400 ms of sustained confidence (real positives hold ≥0.99 much longer).
#ifndef WW_CONSEC
#define WW_CONSEC 2
#endif

// New audio pulled in per poll (samples @16 kHz). 3200 = 200 ms. This is the
// real-time budget for one incremental-feature inference; 1600 (100 ms) OVERRAN
// on the ESP32-S3 with the 32/48/64 CNN and dropped audio (see header). Must
// divide WW_CLIP and be a multiple of WW_FRAME_STEP.
#ifndef WW_HOP
#define WW_HOP 3200
#endif
#if (WW_CLIP % WW_HOP) != 0 || (WW_HOP % WW_FRAME_STEP) != 0
#error "WW_HOP must divide WW_CLIP and be a multiple of WW_FRAME_STEP"
#endif
#define WW_HOPS_PER_WIN   (WW_CLIP / WW_HOP)
#define WW_FRAMES_PER_HOP (WW_HOP / WW_FRAME_STEP)

// Fallback stereo slot for the mic when neither the speech-likeness picker nor
// a past STT capture can decide (only happens on essentially-silent windows,
// which the gate squelches). Slot 1 observed on this board.
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
static int    ww_feat_ch = -1;     // slot the rolling feature buffer was built from
static int    ww_consec = 0;       // consecutive over-threshold polls (debounce)
static float* ww_win[2] = {nullptr, nullptr}; // rolling 1 s window per stereo slot
static float  ww_hp[2][WW_HOPS_PER_WIN];      // per-hop raw peaks (gate = max over ring)
static int    ww_hp_i = 0;         // ring index into ww_hp
static float* ww_featring = nullptr; // rolling log-mel  (WW_FRAMES*WW_MELS)
static float* ww_featx = nullptr;    // inference scratch copy (CMN mutates it)
static float* ww_b1 = nullptr;     // conv1 activations
static float* ww_b2 = nullptr;     // conv2 activations
static float* ww_b3 = nullptr;     // conv3 activations

static inline int ww_oh(int ih){ return (ih-WW_K)/2+1; }

// Allocate scratch (PSRAM) and print model info. Call once at boot.
inline bool wakeWordBegin(){
  int H1=ww_oh(WW_FRAMES), W1=ww_oh(WW_MELS);
  int H2=ww_oh(H1),        W2=ww_oh(W1);
  int H3=ww_oh(H2),        W3=ww_oh(W2);
  ww_win[0]   = (float*)ps_malloc(sizeof(float)*WW_CLIP);
  ww_win[1]   = (float*)ps_malloc(sizeof(float)*WW_CLIP);
  ww_featring = (float*)ps_malloc(sizeof(float)*WW_FRAMES*WW_MELS);
  ww_featx    = (float*)ps_malloc(sizeof(float)*WW_FRAMES*WW_MELS);
  ww_b1       = (float*)ps_malloc(sizeof(float)*H1*W1*WW_C1_OUT);
  ww_b2       = (float*)ps_malloc(sizeof(float)*H2*W2*WW_C2_OUT);
  ww_b3       = (float*)ps_malloc(sizeof(float)*H3*W3*WW_C3_OUT);
  if(!ww_win[0]||!ww_win[1]||!ww_featring||!ww_featx||!ww_b1||!ww_b2||!ww_b3){
    Serial.println("[WakeWord] ❌ PSRAM alloc failed.");
    return false;
  }
  Serial.printf("[WakeWord] 'hey pip' model ready: %d-frame log-mel%s, CNN %d/%d/%d, thr %.2f, hop %d ms\n",
                WW_FRAMES,
#ifdef WW_CMN
                "+CMN",
#else
                "",
#endif
                WW_C1_OUT, WW_C2_OUT, WW_C3_OUT,
                (double)WAKE_WORD_THRESHOLD, WW_HOP/(WW_SR/1000));
  return true;
}

// The .ino calls this after every successful answer capture with the stereo slot
// its (loud-speech, reliable) energy pick chose — ground truth for our picker.
inline void wakeWordNoteSttChannel(int ch){
  if(ch==0 || ch==1) ww_stt_ch = ch;
}

// Begin listening: raise the codec listen gain, install I2S, flush the first DMA
// buffer (codec-settling garbage), clear the windows + feature ring.
inline void wakeWordStartListening(){
  if(ww_active) return;
  es8311SetWakeListenBoost(true);          // +18dB ADC scale-up while we listen
  i2s_start_recording();
  { const int FR=1024; static int16_t tmp[FR*2]; size_t br=0;
    i2s_read(I2S_PORT,(char*)tmp,sizeof(tmp),&br,portMAX_DELAY); }
  memset(ww_win[0],0,sizeof(float)*WW_CLIP);
  memset(ww_win[1],0,sizeof(float)*WW_CLIP);
  memset(ww_hp,0,sizeof(ww_hp));
  ww_hp_i = 0;
  ww_feat_ch = -1;                         // force a one-off full feature rebuild
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

// Slide both rolling windows by WW_HOP, read fresh stereo samples into their
// tails, and record each slot's per-hop RMS/peak (gate ring + test display).
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
  ww_hp[0][ww_hp_i]=p0; ww_hp[1][ww_hp_i]=p1;
  ww_hp_i = (ww_hp_i+1) % WW_HOPS_PER_WIN;
  if(rms0) *rms0 = sqrtf(ss0/(float)WW_HOP);
  if(pk0)  *pk0  = p0;
  if(rms1) *rms1 = sqrtf(ss1/(float)WW_HOP);
  if(pk1)  *pk1  = p1;
}

// RAW peak of the chosen slot's current 1 s window = max of its hop-peak ring.
static float ww_window_peak(int ch){
  float p=0.0f;
  for(int i=0;i<WW_HOPS_PER_WIN;i++) if(ww_hp[ch][i]>p) p=ww_hp[ch][i];
  return p;
}

// SPEECH-LIKENESS of a 1 s window: coefficient of variation of its 50 ms frame
// RMS. Speech modulates strongly (CV >> 0); steady crosstalk doesn't (CV ≈ 0).
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

// Bring the rolling feature buffer up to date for slot `ch`: incremental (only
// the WW_FRAMES_PER_HOP new rows) when the slot is unchanged, full one-off
// rebuild when the picker switched slots (or on the first poll after arming).
static void ww_update_features(int ch){
  if(ch != ww_feat_ch){
    ww_logmel(ww_win[ch], ww_featring);    // ~98 FFTs, one-off (~30 ms) — rare
    ww_feat_ch = ch;
    return;
  }
  memmove(ww_featring, ww_featring + WW_FRAMES_PER_HOP*WW_MELS,
          (WW_FRAMES - WW_FRAMES_PER_HOP)*WW_MELS*sizeof(float));
  for(int f=WW_FRAMES-WW_FRAMES_PER_HOP; f<WW_FRAMES; f++)
    ww_logmel_frame(ww_win[ch], f, &ww_featring[f*WW_MELS]);
}

// Pull one hop, update features, run one inference on the mic slot.
// Returns 1 if "hey pip" detected this poll, 0 if not, -1 if not listening.
inline int wakeWordPoll(){
  if(!ww_active) return -1;
  ww_read_hop(nullptr,nullptr,nullptr,nullptr);

  float cv0 = ww_speechiness(ww_win[0]);
  float cv1 = ww_speechiness(ww_win[1]);
  int   ch  = ww_pick_channel(cv0, cv1);
  ww_last_ch = ch;
  ww_update_features(ch);                  // ALWAYS, even when gated — the ring
                                           // must stay aligned with the window
  if(ww_window_peak(ch) < WW_MIN_PEAK){ ww_consec = 0; return 0; }

  memcpy(ww_featx, ww_featring, sizeof(float)*WW_FRAMES*WW_MELS);
  float prob[WW_NCLASS];
  ww_infer_from_feat(ww_featx, ww_b1, ww_b2, ww_b3, prob);   // prob[1] = "hey pip"
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
// Never returns. One line per hop with everything needed to diagnose an
// unresponsive wake word WITHOUT any cloud setup:
//   dt      = ms this poll took. MUST hover at ~WW_HOP/16 ms (200 default).
//             Consistently higher ⇒ the loop is overrunning and the I2S DMA is
//             DROPPING AUDIO (the model then hears time-compressed garbage and
//             scores ~0.1 on a perfect "hey pip") — raise WW_HOP.
//   p0/p1   = per-hop peak of stereo slot 0 / 1. The one that jumps when you
//             talk is the mic (identical values = codec mirrors both slots).
//   cv0/cv1 = speech-likeness of each slot's window; ch = slot scored.
//   win     = chosen window's RAW peak — compare against WW_MIN_PEAK ([gated]).
//   score   = model confidence for "hey pip"; max = highest so far this run.
// Tuning recipe: say "hey pip" a few times, note 'max', set WAKE_WORD_THRESHOLD
// a touch below it. If 'win' shows [gated] while you speak, lower WW_MIN_PEAK.
inline void wakeWordRunTestMode(){
  Serial.println("\n========== WAKE-WORD TEST MODE ==========");
  Serial.printf ("Threshold=%.2f  gate=%.3f  consec=%d  hop=%d ms  (codec listen boost ACTIVE: +18dB)\n",
                 (double)WAKE_WORD_THRESHOLD, (double)WW_MIN_PEAK, WW_CONSEC, WW_HOP/(WW_SR/1000));
  Serial.println("Say \"hey pip\" and watch 'score'/'max'; 'dt' must stay ~= the hop.");
  Serial.println("This loop never exits; reflash with WW_TEST_MODE 0 for normal use.\n");

  if(!ww_win[0]){
    Serial.println("[WWTEST] scratch not allocated — wakeWordBegin() failed (PSRAM?). Halting.");
    while(true){ faceTick(); delay(200); }
  }

  wakeWordStartListening();
  float maxScore = 0.0f;
  int   tmConsec = 0;
  uint32_t tPrev = millis();
  while(true){
    float r0,p0,r1,p1;
    ww_read_hop(&r0,&p0,&r1,&p1);
    float cv0 = ww_speechiness(ww_win[0]);
    float cv1 = ww_speechiness(ww_win[1]);
    int   ch  = ww_pick_channel(cv0, cv1);
    ww_update_features(ch);
    float winpeak = ww_window_peak(ch);
    memcpy(ww_featx, ww_featring, sizeof(float)*WW_FRAMES*WW_MELS);
    float prob[WW_NCLASS];
    ww_infer_from_feat(ww_featx, ww_b1, ww_b2, ww_b3, prob);
    float score = prob[1];
    if(score > maxScore) maxScore = score;
    // Mirror the live gate + debounce so the display matches real behavior.
    const char* tag = "";
    if(winpeak < WW_MIN_PEAK){ tag = "  [gated: too quiet]"; tmConsec = 0; }
    else if(score >= WAKE_WORD_THRESHOLD){
      if(++tmConsec >= WW_CONSEC) tag = "  <<< FIRE";
      else                        tag = "  (arming)";
    } else { tmConsec = 0; }
    uint32_t tNow = millis();
    Serial.printf("[WWTEST] dt=%3lu p0=%.3f p1=%.3f cv0=%.2f cv1=%.2f ch=%d win=%.3f score=%.3f max=%.3f%s\n",
                  (unsigned long)(tNow-tPrev), (double)p0,(double)p1,
                  (double)cv0,(double)cv1,ch,(double)winpeak,
                  (double)score,(double)maxScore,tag);
    tPrev = tNow;
  }
}
