#pragma once
#include <Wire.h>
#include "pins.h"

// ─────────────────────────────────────────────────────────────────────────────
// ES8311 Audio Codec Driver — derived from official LCD Wiki example
// (Example_17_echo / Example_16_music / es8311.cpp)
//
// MCLK = 384 × 16000 = 6.144 MHz  (I2S_MCLK_MULTIPLE_384 in i2s_start_recording)
// BCLK = 6144000 / (bclk_div=4) = ... actually set by I2S driver automatically
// LRCK = 16000 Hz
// Slot width: 16-bit stereo (32 BCLK per LRCK)
// ─────────────────────────────────────────────────────────────────────────────

static void es8311_write(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(ES8311_I2C_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool initES8311() {
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);

  // Verify codec is on the I2C bus
  Wire.beginTransmission(ES8311_I2C_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.println("[ES8311] Not found on I2C bus! Check wiring.");
    return false;
  }

  // ── Reset ─────────────────────────────────────────────────────────────────
  es8311_write(0x00, 0x1F);   // Soft reset all registers
  delay(20);
  es8311_write(0x00, 0x00);   // Clear reset
  es8311_write(0x00, 0x80);   // ★ Power-on command (was missing — critical!)

  // ── Clock: MCLK pin → 6.144 MHz (384 × 16 kHz) ───────────────────────────
  // Coefficients from official coeff_div table for {6144000, 16000}:
  //   pre_div=3, pre_multi=1, bclk_div=4, lrck=0x00/0xFF, osr=0x10
  es8311_write(0x01, 0x3F);   // ★ Enable ALL clocks, MCLK from MCLK pin (was 0x30)
  es8311_write(0x02, 0x48);   // ★ pre_div=3→(2<<5)=0x40, pre_multi=1→(1<<3)=0x08 (was 0x00)
  es8311_write(0x03, 0x10);   // fs_mode=0, adc_osr=0x10
  es8311_write(0x04, 0x10);   // dac_osr=0x10
  es8311_write(0x05, 0x00);   // adc_div=1, dac_div=1
  es8311_write(0x06, 0x03);   // bclk_div=4 → 4-1=3
  es8311_write(0x07, 0x00);   // lrck_h
  es8311_write(0x08, 0xFF);   // lrck_l

  // ── I2S format: 16-bit (ES8311_RESOLUTION_16 → 3<<2 = 0x0C) ─────────────
  es8311_write(0x09, 0x0C);   // ★ SDP In  (ADC→I2S) 16-bit I2S (was 0x00)
  es8311_write(0x0A, 0x0C);   // ★ SDP Out (I2S→DAC) 16-bit I2S (was 0x00)

  // ── Power up ──────────────────────────────────────────────────────────────
  es8311_write(0x0D, 0x01);   // Power up analog circuitry
  es8311_write(0x0E, 0x02);   // Enable analog PGA + ADC modulator
  es8311_write(0x12, 0x00);   // Power up DAC
  es8311_write(0x13, 0x10);   // ★ Enable output to HP drive (was 0x00)

  // ── Microphone (analog, not PDM) ──────────────────────────────────────────
  es8311_write(0x14, 0x1A);   // Enable analog MIC, DIFFERENTIAL, +PGA (official board value)
                                // This board (per the LCD Wiki ai_chat demo) wires the mic
                                // DIFFERENTIALLY. Single-ended (0x10) recorded only constant
                                // clock/common-mode crosstalk (crest factor ~1.0, level
                                // independent of speech) — differential rejects that and
                                // passes the actual voice. The earlier "differential clips"
                                // result was caused by reg17=0xC8 (+36dB), NOT by MIC1N; with
                                // the low digital gain below it no longer over-drives.
  // ADC gain scale-up (REG16 ADC_SCALE[2:0], datasheet p.20-21): 0..7 -> 0..42dB in
  // 6dB steps, applied INSIDE the codec before the signal is truncated to 16-bit
  // I2S — so unlike software gain it recovers real resolution from the quiet mic.
  // 0x04 (24dB) is the chip's reset default; written explicitly so the level the
  // working STT path was tuned against no longer depends on an implicit default.
  // The wake-word listener temporarily raises this via es8311SetWakeListenBoost().
  es8311_write(0x16, 0x04);   // ADC_SCALE = 24dB (chip default, STT-proven level)

  // ADC DIGITAL volume. With the HPF now removing DC, this gain is clean. At 0xC8 the
  // mic worked but Hebrew speech was quiet (RMS ~928, -31dBFS) and STT returned empty.
  // 0xDA is ~+9dB: targets speech RMS ~2600 / peak ~24500 (-22dBFS) with headroom.
  // Each step = 0.5dB. If loud speech clips (peak hits 32767), step down toward 0xD0.
  es8311_write(0x17, 0xDA);   // ADC digital volume (~+9dB for healthy STT level)

  // ── ADC high-pass filter — REMOVES DC OFFSET (two-stage) ──────────────────
  // CRITICAL: both stages must be written. Stage 1 (0x1B) was previously MISSING,
  // so a DC offset passed through and railed the ADC to a constant value (~8700),
  // with speech only a tiny ripple on top — STT heard no speech. The full
  // Espressif/arduino-audio-driver init writes BOTH of these.
  es8311_write(0x1B, 0x0A);   // ★ ADC HPF stage 1 + ALC automute (was never set)
  es8311_write(0x1C, 0x6A);   // ADC HPF stage 2 + EQ bypass

  // ── DAC equalizer bypass ──────────────────────────────────────────────────
  es8311_write(0x37, 0x08);   // Bypass DAC EQ

  // ── DAC volume (TTS loudness) ─────────────────────────────────────────────
  // Reg 0x32 is the master DAC digital volume: 0.5 dB/step, 0xBF = 0 dB (unity).
  // Bumped 0xBF→0xC5 (+3 dB) so TTS is a bit louder now that the speaker opening is
  // partly covered. Each +2 steps ≈ +1 dB (e.g. 0xC8 ≈ +4.5 dB); back off toward
  // 0xBF if loud TTS distorts — this is pure digital gain, so it can clip source
  // peaks that are already near full-scale (watch the "peak=…/32767" line in tts_player).
  es8311_write(0x32, 0xC5);

  Serial.println("[ES8311] Init OK");
  return true;
}

// ── Wake-word listen gain staging ────────────────────────────────────────────
// While the wake-word listener is armed, raise the codec's internal ADC scale-up
// from the STT-proven 24dB to the maximum 42dB (+18dB). The analog PGA (REG14)
// is already at its 30dB max, so this is the only real gain headroom left — and
// because it scales BEFORE the 16-bit I2S truncation it recovers ~3 bits of true
// resolution from a distant/quiet "hey pip" (raw peak ~0.01 -> ~0.1), instead of
// amplifying quantization noise in software afterwards.
// It is switched OFF again the moment listening stops, so the answer recording /
// STT path always runs at exactly the levels it was tuned for (close speech peaks
// ~0.75 FS at 24dB — +18dB there would clip).
void es8311SetWakeListenBoost(bool on) {
  es8311_write(0x16, on ? 0x07 : 0x04);   // ADC_SCALE: 7=42dB (listen) / 4=24dB (STT)
}
