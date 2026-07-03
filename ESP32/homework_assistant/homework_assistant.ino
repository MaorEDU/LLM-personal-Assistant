/**
 * Homework Assistant — ESP32-S3 Firmware (Emotional Tutor, edge-to-edge)
 * Board: LCDWIKI 2.8" ESP32-S3 Display (ES3C28P / ES3N28P)
 *
 * This firmware drives the team's adaptive tutor loop end-to-end and shows the
 * pip_face animated emotions on the ILI9341 display. It also publishes
 * deviceState so the Flutter parent app sees the device live.
 *
 * Flow:
 *   1. Boot → ES8311 codec → pip_face → WiFi → NTP → Firebase anon auth
 *   2. Create a session (auto-detects the child profile by deviceId)
 *   3. Backend (onSessionCreated) generates the first question + its TTS audio
 *   4. Device SPEAKS the question (face: speaking) then LISTENS (face: listening)
 *   5. Child says the wake word "hey pip" and answers → I2S capture (ended on
 *      silence) → STT proxy.  (Legacy push-to-talk button kept behind
 *      USE_WAKE_WORD=0 — see wake_word.h / Documentation/WAKE_WORD.md.)
 *   6. Device posts a learning_turn → backend grades + Gemini feedback +
 *      next question + combined TTS audio
 *   7. Device shows the returned EMOTION on pip_face, SPEAKS feedback+next
 *      question, then loops back to step 4 with the next question
 *
 * Required libraries (Arduino IDE → Manage Libraries):
 *   - ArduinoJson      by Benoit Blanchon (v7.x)
 *   - TFT_eSPI         by Bodmer            (for pip_face; configure User_Setup)
 *   (Wake word "hey pip" needs NO extra library — the model is bundled in
 *    wake_word.h / ww_infer.h / hey_pip_model.h. See Documentation/WAKE_WORD.md.)
 *   See INTEGRATION.md for the TFT_eSPI User_Setup for this exact board.
 *
 * Board settings (Arduino IDE):
 *   - Board: "ESP32S3 Dev Module"
 *   - Flash Size: 16MB (128Mb)
 *   - PSRAM: "OPI PSRAM"   ← IMPORTANT (audio buffer + pip_face sprite)
 *   - Partition Scheme: "Huge APP"
 *   - Upload Speed: 921600
 */

#include <WiFi.h>
#include <time.h>
#include <driver/i2s.h>   // ESP32-S3 built-in I2S driver
#include "secrets.h"
#include "pins.h"
#include "es8311.h"
// ── pip_face on/off switch ────────────────────────────────────────────────────
// Set to 0 to build & boot WITHOUT the display. The audio tutor loop runs fine
// without it — use this to isolate a display/TFT_eSPI problem from the rest of
// the device. With USE_PIP_FACE=1 you MUST have TFT_eSPI installed AND its
// User_Setup configured for this board (copy pip_face/User_Setup_LCDWIKI.h over
// libraries/TFT_eSPI/User_Setup.h), or TFT_eSPI will crash at init.
#define USE_PIP_FACE 1
#if USE_PIP_FACE
  #include "PipFace.h"     // animated face (TFT_eSPI). See INTEGRATION.md.
#endif
#include "firebase_client.h"
#include "stt_client.h"
#include "tts_player.h"
#include "tts_cache.h"     // SD-backed cache for static/repeated phrases (after the 3 above)

// ── Wake-word ("hey pip") on/off switch ───────────────────────────────────────
// 1 = say "hey pip" to start an answer (REPLACES the button).
// 0 = legacy push-to-talk button (kept intact below; we may still want it).
// The model is fully self-contained (wake_word.h / ww_infer.h / hey_pip_model.h)
// — NOTHING to install. See Documentation/WAKE_WORD.md for how it was trained + tuning.
#ifndef USE_WAKE_WORD
#define USE_WAKE_WORD 1
#endif

// ── Phase 1: collapse the turn into ONE synchronous processTurn call ──────────
// 1 = device calls processTurn (grade + Gemini + TTS + writes done server-side,
//     result returned inline; audio is the HTTP body). Removes the device's own
//     Firestore write, the Eventarc trigger, the poll loop, and the separate
//     audio download. 0 = legacy post+poll path (firestorePostLearningTurn +
//     firestorePollForTurnResult). Flip to 0 to A/B or fall back.
#ifndef USE_PROCESS_TURN
#define USE_PROCESS_TURN 1
#endif

// ── Phase 2: fold STT into processTurn (upload audio; server transcribes) ─────
// 1 = the device uploads the raw mono PCM straight to processTurn (?fmt=pcm16),
//     which runs STT itself, then grade + Gemini + TTS — ONE round trip for the
//     whole turn, with NO separate transcribeAudio handshake and a RAW-PCM upload
//     (no base64, −33% bytes over the slow uplink). Requires USE_PROCESS_TURN.
// 0 = transcribe on-device first (transcribeAudio), then send the TEXT to
//     processTurn (or to post+poll when USE_PROCESS_TURN=0). Flip to 0 to A/B or
//     fall back. The boot identify flow always uses on-device STT regardless.
#ifndef USE_PROCESS_TURN_AUDIO
#define USE_PROCESS_TURN_AUDIO 1
#endif

// ── Wake-word TEST MODE (threshold tuning / "it's unresponsive" debugging) ─────
// Set to 1 to boot straight into a serial score printer: it SKIPS WiFi/Firebase
// and, every ~200 ms, prints the live "hey pip" score plus the mic level so you
// can see exactly why it isn't triggering. Set back to 0 for normal operation.
// How to read the output (Serial Monitor @115200):
//   • rms/peak stay ~0 while you speak  → mic/I2S problem (no audio reaching the model)
//   • score spikes on "hey pip" but stays under the threshold → lower WAKE_WORD_THRESHOLD
//   • score never moves at all          → model/feature problem
// Once you know the score your real "hey pip" hits, set WAKE_WORD_THRESHOLD just
// below it (in wake_word.h), flip this back to 0, and reflash.
#ifndef WW_TEST_MODE
#define WW_TEST_MODE 0
#endif

// ── Wake-word DATA CAPTURE mode (record real samples to SD for RETRAINING) ─────
// Set to 1 to boot straight into the SD recorder (see ww_capture.h): say "hey pip"
// and capture room/bed noise through THIS device's real mic, then retrain the model
// on it. Skips WiFi/Firebase and never returns. Mutually exclusive with normal use
// and with WW_TEST_MODE — reflash with WW_CAPTURE_MODE 0 to go back to the tutor.
#ifndef WW_CAPTURE_MODE
#define WW_CAPTURE_MODE 0
#endif
#if WW_CAPTURE_MODE
  #include "ww_capture.h"     // needs the codec (initES8311) + SD (sdCacheBegin)
#endif

#if USE_WAKE_WORD
  #include "wake_word.h"
#endif

// ── State machine ─────────────────────────────────────────────────────────────
enum State { IDLE, RECORDING, PROCESSING, SPEAKING };
State state = IDLE;

// ── Audio record buffer (PSRAM) ───────────────────────────────────────────────
uint8_t* recordBuf  = nullptr;
size_t   recordBytes = 0;

// ── Tutor loop state ──────────────────────────────────────────────────────────
String g_currentQuestion = "";
// Subject of the active session. Seeded from SESSION_SUBJECT, but the
// kid's voice-pick during identify_subject overwrites the session doc in
// Firestore, and firestoreWaitForCurrentQuestion reads that value back
// into this global so STT/UI logic uses the actual chosen subject.
String g_currentSubject = SESSION_SUBJECT;
int    g_stars           = 0;     // running correct-answer count (shown on strip)
bool   g_haveFace        = false; // pip_face initialised OK

// ── Heartbeat ─────────────────────────────────────────────────────────────────
uint32_t g_lastHeartbeat = 0;
const uint32_t HEARTBEAT_MS = 10000;  // app online-timeout is 30s

// ─────────────────────────────────────────────────────────────────────────────
// I2S setup for RECORDING (raw driver, full-duplex)
// ─────────────────────────────────────────────────────────────────────────────
void i2s_start_recording() {
  i2s_config_t cfg = {
    .mode             = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_RX),
    .sample_rate      = SAMPLE_RATE,
    .bits_per_sample  = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format   = I2S_CHANNEL_FMT_RIGHT_LEFT,  // stereo — matches ES8311's 32-bit frame
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count    = 12,   // 12×512 frames = 384 ms of RX buffering: rides out
                              // heartbeat HTTPS stalls / display work without the
                              // wake-word listener losing audio (drops = the model
                              // hears time-compressed speech and misses).
    .dma_buf_len      = 512,
    .use_apll         = true,
    .tx_desc_auto_clear = true,
    .mclk_multiple    = I2S_MCLK_MULTIPLE_384,
  };
  i2s_pin_config_t pins = {
    .mck_io_num   = PIN_I2S_MCLK,
    .bck_io_num   = PIN_I2S_BCLK,
    .ws_io_num    = PIN_I2S_LRCK,
    .data_out_num = PIN_I2S_DOUT,
    .data_in_num  = PIN_I2S_DIN,
  };
  i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  i2s_set_pin(I2S_PORT, &pins);
  i2s_zero_dma_buffer(I2S_PORT);
}

void i2s_stop_recording() {
  i2s_driver_uninstall(I2S_PORT);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
void connectWiFi() {
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n[WiFi] Connected: " + WiFi.localIP().toString());
  delay(500);
}

void syncClock() {
  // Real time is needed for Firestore timestamps + the app heartbeat freshness.
  configTime(0, 0, NTP_SERVER);     // UTC; gmtime_r() is used downstream
  Serial.print("[Time] Syncing NTP");
  time_t now = time(nullptr);
  uint32_t start = millis();
  while (now < 1000000000 && millis() - start < 10000) {
    delay(300);
    Serial.print(".");
    now = time(nullptr);
  }
  Serial.println(now < 1000000000 ? "  (failed — using placeholder time)"
                                  : "  ok: " + isoNow());
}

// Map the backend emotion vocabulary onto a pip_face emotion label.
const char* pipEmotionFor(const String& e) {
  if (e == "celebrating") return "celebrating";
  if (e == "happy")       return "happy";
  if (e == "encouraging") return "encouraging";
  if (e == "concerned")   return "concerned";
  if (e == "neutral")     return "happy";
  return "encouraging";
}

// These are no-ops when USE_PIP_FACE==0 (or when begin() failed) so the rest of
// the firmware never has to care whether the display is present.
void faceEmotion(const char* label) {
#if USE_PIP_FACE
  if (g_haveFace) Pip::setEmotion(label);
#else
  (void)label;
#endif
}
void faceStatus(const char* status, int mood = -1) {
#if USE_PIP_FACE
  if (g_haveFace) Pip::setDeviceStatus(status, mood);
#else
  (void)status; (void)mood;
#endif
}
void faceStrip(const String& text) {
#if USE_PIP_FACE
  if (g_haveFace) Pip::setStrip(text.c_str(), g_stars);
#else
  (void)text;
#endif
}
void faceTick() {
#if USE_PIP_FACE
  if (g_haveFace) Pip::tick();
#endif
}

// Post-process the audio in recordBuf in-place: pick the channel that
// actually carries the mic, strip to mono, and reject near-silence.
// Returns true if the audio is usable for STT; false if too short / silent.
// recordBytes is updated to the mono-stripped length on success.
bool processRecordedAudio() {
  if (recordBytes < 1000) {
    Serial.println("[Audio] too short, ignoring.");
    return false;
  }
  // Pick the louder stereo channel (the codec is mono on one side).
  int micChannel = 0;
  {
    int16_t* raw = (int16_t*)recordBuf;
    size_t frames = recordBytes / 4;
    int64_t sumSq0 = 0, sumSq1 = 0;
    for (size_t i = 0; i < frames; i++) {
      sumSq0 += (int64_t)raw[i*2]   * raw[i*2];
      sumSq1 += (int64_t)raw[i*2+1] * raw[i*2+1];
    }
    float rms0 = sqrt((float)(sumSq0 / (int64_t)frames));
    float rms1 = sqrt((float)(sumSq1 / (int64_t)frames));
    micChannel = (rms1 > rms0) ? 1 : 0;
    Serial.printf("[Audio] ch0 RMS: %.0f  ch1 RMS: %.0f  -> ch%d\n", rms0, rms1, micChannel);
#if USE_WAKE_WORD
    wakeWordNoteSttChannel(micChannel);   // loud-speech pick = ground truth for the WW slot picker
#endif
  }
  // Strip to mono.
  {
    int16_t* src = (int16_t*)recordBuf;
    int16_t* dst = (int16_t*)recordBuf;
    size_t frames = recordBytes / 4;
    for (size_t i = 0; i < frames; i++) dst[i] = src[i * 2 + micChannel];
    recordBytes = frames * 2;
  }
  // Silence reject.
  {
    int16_t* samples = (int16_t*)recordBuf;
    int64_t sumSq = 0; size_t n = recordBytes / 2;
    for (size_t i = 0; i < n; i++) sumSq += (int64_t)samples[i] * samples[i];
    float rms = sqrt((float)(sumSq / (int64_t)n));
    Serial.printf("[Audio] mono RMS: %.0f\n", rms);
    if (rms < 250) {
      Serial.println("[Audio] near-silence; speak closer.");
      return false;
    }
  }
  return true;
}

// RMS above which a chunk counts as speech (computed over the interleaved
// stereo stream — the silent codec channel halves it, so this sits a bit below
// the mono speech levels the ES8311 produces). Used for silence endpointing.
#define VOICE_RMS_THRESHOLD 300

// Record an answer that ENDS ON SILENCE (the wake-word replacement for "record
// while the button is held"). Fills recordBuf with stereo PCM and sets
// recordBytes, exactly like the button capture, so processRecordedAudio() /
// processCapturedAnswer() handle it unchanged. Installs and uninstalls I2S
// itself. Returns true if speech was captured, false on timeout/too-short.
//   startTimeoutMs : how long to wait for the child to START talking
//   trailSilenceMs : how much trailing quiet ends the utterance. Lowered 1200→800
//     to shave perceived end-of-speech delay (Phase 0.5). TUNE THIS: too low and
//     a child who pauses mid-answer gets cut off — raise it back toward 1000–1200
//     if you see premature endpoints in the logs.
bool recordUntilSilence(uint32_t startTimeoutMs = 6000, uint32_t trailSilenceMs = 800) {
  i2s_start_recording();
  recordBytes = 0;
  uint32_t startMs    = millis();
  uint32_t lastVoice  = startMs;
  bool     heardVoice = false;
  const size_t CHUNK  = 1024;   // bytes per i2s_read (256 stereo frames)

  while (recordBytes + CHUNK <= RECORD_BUF_SIZE) {
    size_t br = 0;
    i2s_read(I2S_PORT, recordBuf + recordBytes, CHUNK, &br, portMAX_DELAY);

    // RMS of this chunk (both channels) → voice-activity decision.
    int16_t* s = (int16_t*)(recordBuf + recordBytes);
    size_t n = br / 2;
    int64_t sumSq = 0;
    for (size_t i = 0; i < n; i++) sumSq += (int64_t)s[i] * s[i];
    float rms = (n > 0) ? sqrt((float)(sumSq / (int64_t)n)) : 0;
    recordBytes += br;

    uint32_t now = millis();
    if (rms > VOICE_RMS_THRESHOLD) { heardVoice = true; lastVoice = now; }
    faceTick();

    if (heardVoice && (now - lastVoice) > trailSilenceMs) break;        // end of speech
    if (!heardVoice && (now - startMs) > startTimeoutMs) break;         // nobody spoke
    if ((now - startMs) > MAX_RECORD_MS) break;                          // hard cap
  }
  i2s_stop_recording();
  Serial.printf("[Rec] %u bytes (%.1f s)%s\n", recordBytes,
                recordBytes / (float)(SAMPLE_RATE * 2 * 2),
                heardVoice ? "" : " — no speech");
  return heardVoice && recordBytes > 1000;
}

// Blocking record: face listens, waits for the trigger, captures the answer.
// In wake-word mode the trigger is the Hebrew wake word and capture ends on
// silence; in button mode it is the original push-to-talk. Used by the
// boot-time identify flow before the main state-machine loop takes over.
bool recordOneAnswerBlocking(uint32_t maxWaitMs = 20000) {
  faceEmotion("listening");
#if USE_WAKE_WORD
  // Wake-word trigger: wait for "היי פיפ", then record until the child stops.
  Serial.println("[Identify] Say \"היי פיפ\", then answer...");
  wakeWordStartListening();
  uint32_t waitStart = millis();
  bool triggered = false;
  while (millis() - waitStart < maxWaitMs) {
    if (wakeWordPoll() == 1) { triggered = true; break; }
    faceTick();
  }
  wakeWordStopListening();
  if (!triggered) {
    Serial.println("[Identify] No wake word within timeout.");
    return false;
  }
  Serial.println("[Identify] Recording...");
  return recordUntilSilence();
#else
  // ── Legacy push-to-talk (kept; we may still want the button) ──────────────
  Serial.println("[Identify] Press the button and answer...");
  uint32_t waitStart = millis();
  while (digitalRead(PIN_BTN) != LOW) {
    faceTick();
    if (millis() - waitStart > maxWaitMs) {
      Serial.println("[Identify] No button press within timeout.");
      return false;
    }
    delay(20);
  }
  Serial.println("[Identify] Recording...");
  recordBytes = 0;
  i2s_start_recording();
  while (digitalRead(PIN_BTN) == LOW && recordBytes < RECORD_BUF_SIZE) {
    size_t bytesRead = 0;
    i2s_read(I2S_PORT,
             recordBuf + recordBytes,
             min((size_t)1024, RECORD_BUF_SIZE - recordBytes),
             &bytesRead, portMAX_DELAY);
    recordBytes += bytesRead;
  }
  i2s_stop_recording();
  Serial.printf("[Identify] Recorded %u bytes (%.1f s)\n",
                recordBytes, recordBytes / (float)(SAMPLE_RATE * 2));
  return true;
#endif
}

// Capture one spoken answer during the boot identify flow, RE-ASKING until the
// child actually says something the STT can transcribe. Never returns empty: it
// keeps re-prompting (with `retryPrompt`) and listening until it has a
// transcript, so the caller never has to fall back to the generic session just
// because the child was slow, silent, or didn't trigger the wake word.
String identifyCaptureAnswer(const char* retryPrompt) {
  while (true) {
    if (recordOneAnswerBlocking() && processRecordedAudio()) {
      faceEmotion("thinking");
      String t = transcribeAudio(recordBuf, recordBytes, g_idToken, "he-IL");
      if (!t.isEmpty()) return t;
    }
    // No trigger, silence, or unintelligible speech — re-ask and listen again.
    // retryPrompt is a fixed string → SD cache makes the re-ask instant.
    faceEmotion("encouraging");
    speakTextCached(retryPrompt);
  }
}

// Boot-time voice identification flow:
//   1. Robot asks "מי כאן?" → kid speaks name → identify_child exchange
//   2. Robot replies "שלום X, מה נלמד היום?" → kid says "חשבון" or "אנגלית"
//      → identify_subject exchange
//   3. The identify_subject reply also contains the FIRST question — so
//      after this returns, the main loop is ready to start practising.
// Returns true on success; on failure the caller can fall back to the
// legacy "awaitingFirstQuestion" flow.
bool runIdentifyFlow(String& firstQuestionOut, String& firstAudioUrlOut) {
  // ── Step 1 — ask "who's here?" ────────────────────────────────────────────
#if USE_WAKE_WORD
  const char* welcomeText = "היי! מי כאן? תגיד \"היי פיפ\" ואז את השם שלך.";
#else
  const char* welcomeText = "היי! מי כאן? תגיד לי את שמך אחרי שתלחץ על הכפתור.";
#endif
  faceEmotion("speaking");
  if (!speakTextCached(welcomeText)) {     // SD cache → instant after first boot
    Serial.println("[Identify] TTS welcome failed.");
    return false;
  }

  // Repeat "who's here?" until the cloud matches a real child — never fall back
  // to a generic session just because the child was slow, silent, or
  // mis-recognised. The only non-retry exit is needsPairing: the device's UID
  // isn't linked to any parent, so no spoken name can ever match and looping
  // would trap the child forever.
#if USE_WAKE_WORD
  const char* nameRetry = "לא שמעתי. תגיד \"היי פיפ\" ואז את השם שלך שוב.";
#else
  const char* nameRetry = "לא שמעתי. תלחץ על הכפתור ותגיד שוב את שמך.";
#endif
  IdentifyResult child;
  while (true) {
    String nameTranscript = identifyCaptureAnswer(nameRetry);

    String childExchange = firestorePostIdentifyChild(g_sessionId, nameTranscript);
    if (childExchange.isEmpty() ||
        !firestorePollForIdentifyResult(g_sessionId, childExchange, child)) {
      delay(1000);   // transient cloud error — pause, then ask again
      continue;
    }
    Serial.printf("[Identify] matched: %s (%s)\n",
                  child.matchedChildName.c_str(), child.matchedChildId.c_str());
    if (child.matchedChildId.length() > 0) break;            // success
    if (child.needsPairing) {
      if (!child.audioUrl.isEmpty()) { faceEmotion("speaking"); speakAudio(child.audioUrl); }
      Serial.println("[Identify] device not paired — cannot identify; falling back.");
      return false;
    }
    // Name not recognised — re-ask and keep looping (cached static phrase).
    faceEmotion("encouraging");
    speakTextCached("לא הכרתי את השם הזה. תגיד אותו שוב, בבקשה.");
  }
  g_childId = child.matchedChildId;

  // ── Step 2 — speak greeting + ask for subject, record answer ──────────────
  if (!child.audioUrl.isEmpty()) {
    faceEmotion("speaking");
    speakAudio(child.audioUrl);
  }

  // Repeat "math or english?" until the cloud returns a recognised subject
  // (which also carries the first question). Same rule as the name step: keep
  // asking rather than dropping into a generic session on a missed answer.
  IdentifyResult subj;
  while (true) {
    String subjectTranscript = identifyCaptureAnswer("לא שמעתי. חשבון או אנגלית?");

    String subjectExchange = firestorePostIdentifySubject(g_sessionId, subjectTranscript);
    if (subjectExchange.isEmpty() ||
        !firestorePollForIdentifyResult(g_sessionId, subjectExchange, subj)) {
      delay(1000);   // transient cloud error — pause, then ask again
      continue;
    }
    if (subj.subject.length() > 0) break;                    // success
    // Subject not understood — re-ask and keep looping (cached static phrase).
    faceEmotion("encouraging");
    speakTextCached("לא הבנתי. תגיד חשבון או אנגלית?");
  }
  Serial.printf("[Identify] subject: %s\n", subj.subject.c_str());
  g_currentSubject  = subj.subject;
  firstQuestionOut  = subj.nextQuestion;
  firstAudioUrlOut  = subj.audioUrl;
  return true;
}

// Speak the current question and move into the "waiting for the child" state.
void askCurrentQuestion(const String& audioUrl) {
  Serial.println("[Tutor] Question: " + g_currentQuestion);
  faceStatus("asking");
  faceStrip(g_currentQuestion);
  firestoreWriteDeviceState("asking", g_currentQuestion);
  faceTick();

  state = SPEAKING;
  speakAudio(audioUrl);            // releases I2S when done

  // Now listen.
  state = IDLE;
  faceStatus("listening");
  firestoreWriteDeviceState("listening", g_currentQuestion);
#if USE_WAKE_WORD
  Serial.println("\n🪄 Say \"היי פיפ\" then your answer.");
  wakeWordStartListening();        // begin continuous wake-word detection
#else
  Serial.println("\n🎤 Hold the button and say your answer.");
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== Homework Assistant (Emotional Tutor) Booting ===");

  // Button — COMMENTED OUT: the Hebrew wake word "היי פיפ" now triggers
  // recording (see USE_WAKE_WORD). Kept here, compiled only when the wake word
  // is disabled, so we can fall back to push-to-talk if we ever need it.
#if !USE_WAKE_WORD
  // IO3 reads press, IO2 acts as GND (OUTPUT LOW).
  // (If the IO3 expansion line is flaky on your unit, switch PIN_BTN to GPIO0 /
  //  the onboard BOOT button in pins.h — see INTEGRATION.md.)
  pinMode(PIN_BTN_GND, OUTPUT);
  digitalWrite(PIN_BTN_GND, LOW);
  pinMode(PIN_BTN, INPUT_PULLUP);
#endif

  // Speaker amp — disabled until needed.
  pinMode(PIN_AUDIO_EN, OUTPUT);
  digitalWrite(PIN_AUDIO_EN, HIGH);

  // Display backlight (pip_face / TFT_eSPI does not manage it).
  pinMode(PIN_LCD_BL, OUTPUT);
  digitalWrite(PIN_LCD_BL, HIGH);

  // PSRAM check + record buffer.
  if (!psramFound()) {
    Serial.println("⚠️  PSRAM not found! Check board settings (PSRAM: OPI PSRAM).");
  } else {
    recordBuf = (uint8_t*)ps_malloc(RECORD_BUF_SIZE);
    Serial.printf("[PSRAM] Record buffer: %u bytes\n", RECORD_BUF_SIZE);
  }

  // SD card: enables the TTS cache for static/repeated phrases. Independent of
  // Wi-Fi; fails safe (caching off, cloud fallback) if no FAT32 card is present.
  sdCacheBegin();

  // pip_face (needs PSRAM for its 240x240 sprite, and a correct TFT_eSPI
  // User_Setup for this board).
#if USE_PIP_FACE
  g_haveFace = Pip::begin();
  if (!g_haveFace) Serial.println("⚠️  pip_face init failed (PSRAM/TFT_eSPI). Continuing audio-only.");
  faceEmotion("thinking");
  faceTick();
#else
  Serial.println("[pip_face] disabled at build time (USE_PIP_FACE=0).");
#endif

  // ES8311 codec via I2C.
  if (!initES8311()) Serial.println("⚠️  ES8311 init failed. Audio may not work.");

#if WW_CAPTURE_MODE
  // Data-collection build: record real "hey pip" + noise to the SD card, then halt.
  // Needs the codec (just above) and the SD card (sdCacheBegin, above). Never returns.
  wakeWordRunCaptureMode();
#endif

#if USE_WAKE_WORD
  // Self-contained "hey pip" model (no Edge Impulse). Needs the codec up (shares the mic).
  if (!wakeWordBegin())
    Serial.println("⚠️  Wake-word model not ready — PSRAM allocation failed (enable OPI PSRAM).");
  #if WW_TEST_MODE
  // Diagnostic build: stream the live score for threshold tuning. Never returns,
  // so we skip WiFi/Firebase entirely — just open the Serial Monitor @115200.
  wakeWordRunTestMode();
  #endif
#endif

  // Network + time + auth.
  connectWiFi();
  syncClock();

  // Boot-time auth: restores the saved anonymous UID from NVS when possible so
  // a `children.deviceId` paired via the Flutter app keeps matching across
  // device reboots. Falls back to a fresh signUp on the very first boot or if
  // the saved refresh token has been invalidated.
  g_idToken = firebaseBootAuth();
  if (g_idToken.isEmpty()) {
    Serial.println("❌ Firebase auth failed. Check FIREBASE_WEB_API_KEY in secrets.h");
    faceStatus("error");
    while (true) { faceTick(); delay(200); }
  }

  // Publish the user-visible pairing code → this device's UID so the parent app
  // can pair against the right Firebase identity. Cheap (one PATCH) and idempotent.
  firestoreWritePairingCode();

  // Prefetch the FIXED phrases (boot + reprompts) into the SD cache so their first
  // use is an instant hit, not a ~2–3 s synth. Strings here MUST match the ones
  // passed to speakTextCached() below verbatim (same bytes → same cache key). Runs
  // once; on later boots every phrase is already on the card and is skipped.
  static const char* const STATIC_TTS_PHRASES[] = {
#if USE_WAKE_WORD
    "היי! מי כאן? תגיד \"היי פיפ\" ואז את השם שלך.",
    "לא שמעתי. תגיד \"היי פיפ\" ואז את השם שלך שוב.",
    "לא שמעתי אותך. תגיד \"היי פיפ\" ותענה שוב.",
#else
    "היי! מי כאן? תגיד לי את שמך אחרי שתלחץ על הכפתור.",
    "לא שמעתי. תלחץ על הכפתור ותגיד שוב את שמך.",
    "לא שמעתי אותך. נסה לענות שוב.",
#endif
    "לא הכרתי את השם הזה. תגיד אותו שוב, בבקשה.",
    "לא שמעתי. חשבון או אנגלית?",
    "לא הבנתי. תגיד חשבון או אנגלית?",
  };
  sdCacheWarm(STATIC_TTS_PHRASES,
              sizeof(STATIC_TTS_PHRASES) / sizeof(STATIC_TTS_PHRASES[0]));

  // Two boot paths:
  //   A) CHILD_ID hardcoded → legacy fast path: cloud auto-creates the first
  //      question via onSessionCreated(awaitingFirstQuestion:true).
  //   B) CHILD_ID empty     → voice identification flow:
  //      "מי כאן?" → identify_child → "חשבון או אנגלית?" → identify_subject
  //      → first question is created by handleIdentifySubject.
  const bool useIdentifyFlow = (String(CHILD_ID).length() == 0);

  g_sessionId = firestoreCreateSession(SESSION_SUBJECT, !useIdentifyFlow);
  if (g_sessionId.isEmpty()) {
    Serial.println("❌ Could not create Firestore session.");
    faceStatus("error");
    firestoreWriteDeviceState("error");
    while (true) { faceTick(); delay(200); }
  }
  firestoreWriteDeviceState("idle");

  String firstAudio;
  bool ready = false;

  if (useIdentifyFlow) {
    Serial.println("[Tutor] Running voice identification flow...");
    ready = runIdentifyFlow(g_currentQuestion, firstAudio);
    if (!ready) {
      Serial.println("⚠️  Identify flow failed — falling back to legacy generic-session path.");
      // Recreate the session in legacy mode so onSessionCreated kicks in.
      g_sessionId = firestoreCreateSession(SESSION_SUBJECT, true);
    }
  }
  if (!ready) {
    Serial.println("[Tutor] Waiting for the first question from the backend...");
    faceEmotion("thinking");
    ready = firestoreWaitForCurrentQuestion(g_sessionId, g_currentQuestion, firstAudio, 30000, &g_currentSubject);
  }
  if (!ready) {
    Serial.println("❌ No first question. Check Cloud Functions / Vertex AI setup.");
    faceStatus("error");
    firestoreWriteDeviceState("error");
    while (true) { faceTick(); delay(200); }
  }

  askCurrentQuestion(firstAudio);
}

// ─────────────────────────────────────────────────────────────────────────────
// Return to the IDLE "waiting for the child" state. In wake-word mode this also
// re-arms continuous "היי פיפ" detection, so the device is never left deaf after
// a turn or an early-out (empty STT, silence, backend hiccup).
void backToListening() {
  state = IDLE;
  faceStatus("listening");
#if USE_WAKE_WORD
  wakeWordStartListening();
#endif
}

// A capture didn't yield a usable answer (empty STT or near-silence). Re-speak
// the CURRENT question so the child knows what to answer — not just a generic
// "try again" — then re-arm listening so the device never goes silently "stuck".
// In wake-word mode also remind them to say "היי פיפ" first. The full question
// is also already shown on the strip via askCurrentQuestion().
void repromptAfterMiss() {
  String q = g_currentQuestion;
  q.trim();
#if USE_WAKE_WORD
  const char* tail = "תגיד \"היי פיפ\" ותענה שוב.";
#else
  const char* tail = "נסה לענות שוב.";
#endif
  // STATIC reprompt (no embedded question) so it's one fixed, prefetchable phrase
  // — an instant SD hit instead of a unique per-question synth every miss. The
  // question itself is already shown on the strip via faceStrip(), so the child
  // still sees what to answer.
  (void)q;   // kept for context/logging; intentionally not spoken
  String msg = String("לא שמעתי אותך. ") + tail;
  faceEmotion("encouraging");
  speakTextCached(msg);
  backToListening();
}

// Process whatever is in recordBuf (stereo PCM, recordBytes long): strip to the
// mic channel, reject silence, run STT, post the learning turn, speak feedback +
// next question, then go back to listening. Shared by the wake-word path and the
// legacy button path so the heavy logic lives in exactly one place.
void processCapturedAnswer() {
  if (recordBytes < 1000) {
    Serial.println("[Main] Recording too short, ignoring.");
    backToListening();
    return;
  }

  // ── Pick the stereo slot that actually carries the mic (mono codec) ────────
  int micChannel = 0;
  {
    int16_t* raw = (int16_t*)recordBuf;
    size_t frames = recordBytes / 4;
    int64_t sumSq0 = 0, sumSq1 = 0;
    for (size_t i = 0; i < frames; i++) {
      sumSq0 += (int64_t)raw[i*2]   * raw[i*2];
      sumSq1 += (int64_t)raw[i*2+1] * raw[i*2+1];
    }
    float rms0 = sqrt((float)(sumSq0 / (int64_t)frames));
    float rms1 = sqrt((float)(sumSq1 / (int64_t)frames));
    micChannel = (rms1 > rms0) ? 1 : 0;
    Serial.printf("[Main] ch0 RMS: %.0f  ch1 RMS: %.0f  -> ch%d\n", rms0, rms1, micChannel);
#if USE_WAKE_WORD
    wakeWordNoteSttChannel(micChannel);   // loud-speech pick = ground truth for the WW slot picker
#endif
  }
  // Strip to mono (keep the mic channel).
  {
    int16_t* src = (int16_t*)recordBuf;
    int16_t* dst = (int16_t*)recordBuf;
    size_t frames = recordBytes / 4;
    for (size_t i = 0; i < frames; i++) dst[i] = src[i * 2 + micChannel];
    recordBytes = frames * 2;
  }

  // Reject near-silence before paying for STT (STT hallucinates on noise).
  {
    int16_t* samples = (int16_t*)recordBuf;
    int64_t sumSq = 0; size_t n = recordBytes / 2;
    for (size_t i = 0; i < n; i++) sumSq += (int64_t)samples[i] * samples[i];
    float rms = sqrt((float)(sumSq / (int64_t)n));
    Serial.printf("[Main] Answer RMS: %.0f\n", rms);
    if (rms < 250) {
      Serial.println("[Main] ⚠️  Near-silence — skipping. Speak close to the mic.");
      repromptAfterMiss();   // re-speak the question so the child knows what to answer
      return;
    }
  }

  // STT — pick language by the SESSION's subject (read from Firestore after
  // identify_subject), not the compile-time SESSION_SUBJECT. So an English
  // lesson is recognised as English even when flashed with SESSION_SUBJECT="math".
  // [lat] t0 = child finished recording (entry to processing). All deltas below
  // are measured against this and against each preceding stage.
  uint32_t _latT0 = millis();

  faceEmotion("thinking");
  faceTick();
  const char* sttLang = g_currentSubject.equalsIgnoreCase("english") ? "en-US" : "he-IL";

  TurnResult turn;

#if USE_PROCESS_TURN && USE_PROCESS_TURN_AUDIO
  // Phase 2: upload the raw mono PCM straight to processTurn. ONE round trip does
  // STT + grade + (gated) Gemini + TTS + the doc writes, returned inline (metadata
  // in headers, MP3 in the body). Removes the separate transcribeAudio handshake
  // and the base64 upload overhead entirely.
  uint32_t _latPreTurn = millis();
  if (!firestoreProcessTurnAudio(g_sessionId, recordBuf, recordBytes, sttLang, turn)) {
    backToListening();
    return;
  }
  uint32_t _latPostTurn = millis();
  if (turn.sttEmpty) {
    Serial.println("[Main] Server STT found no speech — asking child to repeat.");
    repromptAfterMiss();   // re-speak the question so the child knows what to answer
    return;
  }
  Serial.printf("[lat]   A->E processTurnAudio(stt+turn)=%lu ms\n",
                (unsigned long)(_latPostTurn - _latPreTurn));
#else
  // On-device STT first, then send the TEXT onward. (Phase 1, or legacy post+poll
  // when USE_PROCESS_TURN=0.)
  uint32_t _latPreStt = millis();   // after local pre-processing (mono strip, RMS gate)
  String answer = transcribeAudio(recordBuf, recordBytes, g_idToken, sttLang);
  uint32_t _latPostStt = millis();
  Serial.printf("[lat]   B stt=%lu ms\n", (unsigned long)(_latPostStt - _latPreStt));
  if (answer.isEmpty()) {
    Serial.println("[Main] STT returned empty transcript — asking child to repeat.");
    repromptAfterMiss();   // re-speak the question so the child knows what to answer
    return;
  }
  Serial.println("[Main] Child answered: " + answer);

  #if USE_PROCESS_TURN
  // Phase 1: ONE synchronous call does grade + (gated) Gemini + TTS + the doc
  // writes, and returns the result inline (metadata in headers, MP3 in the body).
  // Replaces the device write + Eventarc trigger + poll loop + separate download.
  uint32_t _latPreTurn = millis();
  if (!firestoreProcessTurn(g_sessionId, answer, turn)) {
    backToListening();
    return;
  }
  uint32_t _latPostTurn = millis();
  Serial.printf("[lat]   B->E processTurn=%lu ms\n", (unsigned long)(_latPostTurn - _latPreTurn));
  #else
  // Legacy post + poll path (kept as fallback; flip USE_PROCESS_TURN to 0).
  uint32_t _latPrePost = millis();
  String exchangeId = firestorePostLearningTurn(g_sessionId, answer);
  uint32_t _latPostPost = millis();
  Serial.printf("[lat]   C post=%lu ms\n", (unsigned long)(_latPostPost - _latPrePost));
  if (exchangeId.isEmpty()) { backToListening(); return; }

  uint32_t _latPrePoll = millis();
  if (!firestorePollForTurnResult(g_sessionId, exchangeId, turn, 30000)) {
    backToListening();
    return;
  }
  uint32_t _latPostPoll = millis();
  Serial.printf("[lat]   D+E+F trigger+compute+poll=%lu ms\n",
                (unsigned long)(_latPostPoll - _latPrePoll));
  #endif
#endif

  if (turn.isCorrect) g_stars++;
  Serial.println("[Main] Feedback: " + turn.spokenFeedback);
  Serial.println("[Main] Next:     " + turn.nextQuestion);
  Serial.println("[Main] Emotion:  " + turn.emotion);

  // Show emotion + speak feedback (audio already includes the next question).
  faceEmotion(pipEmotionFor(turn.emotion));
  // (No deviceState write here — the post-playback "listening" write below pushes
  // the next question; keeping a write on this line only delayed playback ~2 s.)
  faceStrip(turn.nextQuestion);

  state = SPEAKING;
  uint32_t _latPrePlay = millis();
#if USE_PROCESS_TURN
  speakFromBuffer(turn.audioBuf, turn.audioLen);   // inline MP3 from the turn response
  if (turn.audioBuf) { free(turn.audioBuf); turn.audioBuf = nullptr; }
#else
  speakAudio(turn.audioUrl);          // installs its own I2S; releases it when done
#endif

  // [lat] One-line round-trip summary. g_ttsFirstSampleMs was stamped the instant
  // the first audio sample hit I2S (robot starts speaking).
  uint32_t _latFirstSound = g_ttsFirstSampleMs ? g_ttsFirstSampleMs : millis();
#if USE_PROCESS_TURN && USE_PROCESS_TURN_AUDIO
  // Phase 2: STT is folded into processTurnAudio, so there's no separate stt stage.
  Serial.printf(
    "[lat] === turn: processTurnAudio(stt+turn)=%lu decode=%lu "
    "| RECORD->FIRST_SOUND=%lu ms ===\n",
    (unsigned long)(_latPostTurn - _latPreTurn),
    (unsigned long)(_latFirstSound - _latPrePlay),
    (unsigned long)(_latFirstSound - _latT0));
#elif USE_PROCESS_TURN
  Serial.printf(
    "[lat] === turn: stt=%lu processTurn=%lu decode=%lu "
    "| RECORD->FIRST_SOUND=%lu ms ===\n",
    (unsigned long)(_latPostStt  - _latPreStt),
    (unsigned long)(_latPostTurn - _latPreTurn),
    (unsigned long)(_latFirstSound - _latPrePlay),
    (unsigned long)(_latFirstSound - _latT0));
#else
  Serial.printf(
    "[lat] === turn: stt=%lu post=%lu trig+compute+poll=%lu audio(G)=%lu "
    "| RECORD->FIRST_SOUND=%lu ms ===\n",
    (unsigned long)(_latPostStt  - _latPreStt),
    (unsigned long)(_latPostPost - _latPrePost),
    (unsigned long)(_latPostPoll - _latPrePoll),
    (unsigned long)(_latFirstSound - _latPrePlay),
    (unsigned long)(_latFirstSound - _latT0));
#endif

  // Session ended explicitly (exit intent / continue declined / 50-min limit).
  if (turn.sessionEnded) {
    Serial.println("[Main] Session ended by child or time limit. Restarting in 3s...");
    faceStatus("idle");
    firestoreWriteDeviceState("idle");
    delay(3000);
    ESP.restart();   // fresh boot → new session + re-identification
    return;
  }

  // The next question is now current.
  g_currentQuestion = turn.nextQuestion;

  if (turn.shouldTakeBreak) {
    Serial.println("[Main] Taking a short break.");
    faceStatus("break");
    firestoreWriteDeviceState("break", g_currentQuestion);
    faceTick();
    delay(4000);
  }

  // Ready for the next answer.
  firestoreWriteDeviceState("listening", g_currentQuestion);
#if USE_WAKE_WORD
  Serial.println("\n🪄 Say \"היי פיפ\" then your answer.");
#else
  Serial.println("\n🎤 Hold the button and say your answer.");
#endif
  backToListening();                  // re-arms wake-word detection in wake mode
}

// ─────────────────────────────────────────────────────────────────────────────
void loop() {
  faceTick();  // ~30fps internally; cheap to call every iteration

  // Heartbeat so the app keeps showing the device online. Uses the keep-alive
  // path (firestoreHeartbeat) so repeats cost ~0.1 s instead of a ~2 s handshake —
  // the device stays responsive to "היי פיפ" between heartbeats. Meaningful state
  // transitions still use firestoreWriteDeviceState().
  if (millis() - g_lastHeartbeat > HEARTBEAT_MS) {
    g_lastHeartbeat = millis();
    const char* hb = (state == IDLE) ? "listening" : "asking";
    firestoreHeartbeat(hb, g_currentQuestion);
  }

#if USE_WAKE_WORD
  // ── IDLE → listen for the Hebrew wake word "היי פיפ" ───────────────────────
  // Replaces the button: each loop iteration feeds one audio slice to the Edge
  // Impulse classifier. On a hit we hand I2S to the recorder, capture the answer
  // (ended by silence), and process it through the shared pipeline.
  if (state == IDLE) {
    if (wakeWordPoll() == 1) {
      Serial.println("[Main] 🪄 \"היי פיפ\" detected — listening for the answer.");
      wakeWordStopListening();          // release I2S so the recorder can install it
      faceEmotion("listening");
      state = RECORDING;
      if (recordUntilSilence()) {
        state = PROCESSING;
        processCapturedAnswer();        // returns to IDLE + re-arms the wake word
      } else {
        Serial.println("[Main] No speech after the wake word.");
        backToListening();
      }
    }
  }
#else
  // ── Legacy push-to-talk (kept; we may still want the button) ───────────────
  bool btnDown = (digitalRead(PIN_BTN) == LOW);

  // IDLE → start recording the child's answer when the button is pressed.
  if (state == IDLE && btnDown) {
    Serial.println("[Main] Recording answer...");
    state = RECORDING;
    recordBytes = 0;
    faceEmotion("listening");
    i2s_start_recording();
  }

  // RECORDING → capture while the button is held.
  if (state == RECORDING) {
    if (btnDown && recordBytes < RECORD_BUF_SIZE) {
      size_t bytesRead = 0;
      i2s_read(I2S_PORT,
               recordBuf + recordBytes,
               min((size_t)1024, RECORD_BUF_SIZE - recordBytes),
               &bytesRead, portMAX_DELAY);
      recordBytes += bytesRead;
      if (recordBytes >= RECORD_BUF_SIZE) {
        Serial.println("[Main] Buffer full — stopping recording.");
        btnDown = false;
      }
    }

    if (!btnDown) {  // button released → process this answer
      i2s_stop_recording();
      state = PROCESSING;
      Serial.printf("[Main] Recorded %u bytes (%.1f sec)\n",
                    recordBytes, recordBytes / (float)(SAMPLE_RATE * 2));
      processCapturedAnswer();          // shared pipeline (same as wake-word path)
    }
  }
#endif
}
