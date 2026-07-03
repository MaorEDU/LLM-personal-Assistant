#pragma once
// Board: LCDWIKI 2.8" ESP32-S3 Display (ES3C28P / ES3N28P)
// See ESP32/parameters.h for the full annotated pin table.

// ── Audio (ES8311 codec) ─────────────────────────────────────────────────────
#define PIN_AUDIO_EN    1    // LOW = speaker amp on
#define PIN_I2S_MCLK    4
#define PIN_I2S_BCLK    5
#define PIN_I2S_DOUT    8    // ESP32 → speaker  (confirmed from LCD Wiki echo example)
#define PIN_I2S_LRCK    7
#define PIN_I2S_DIN     6    // Microphone → ESP32 (confirmed from LCD Wiki echo example)

// ── ES8311 I2C ───────────────────────────────────────────────────────────────
#define PIN_I2C_SDA     16
#define PIN_I2C_SCL     15
#define ES8311_I2C_ADDR 0x18

// ── Button (push-to-talk, on IO2/IO3/IO14/IO21 expansion header) ─────────────
#define PIN_BTN         3    // IO3 — INPUT_PULLUP, active LOW
#define PIN_BTN_GND     2    // IO2 — set OUTPUT LOW to act as GND for the button
                             // Wire button between IO2 and IO3 on the expansion header.
                             // IO43/IO44 (UART) can't be used: TX idles HIGH, RX is
                             // held by UART peripheral.
//#define PIN_BTN       0    // (BOOT button fallback — GPIO0, if IO3 line is broken)

// ── RGB status LED (WS2812) ──────────────────────────────────────────────────
#define PIN_RGB_LED     42

// ── Display: ILI9341 SPI (driven by pip_face via TFT_eSPI) ───────────────────
// These do NOT collide with the I2S audio pins (4,5,6,7,8), the I2C codec/touch
// bus (15,16) or the button (2,3) — pip_face and the audio path coexist.
// TFT_eSPI itself is configured in its User_Setup (see pip_face/User_Setup_LCDWIKI.h).
// The backlight is NOT managed by TFT_eSPI/pip_face, so the firmware drives it.
#define PIN_LCD_BL      45   // Backlight enable (HIGH = on)
//      PIN_LCD_CS      10
//      PIN_LCD_MOSI    11
//      PIN_LCD_SCLK    12
//      PIN_LCD_MISO    13
//      PIN_LCD_DC      46
//      PIN_LCD_RST     -1   // tied to board reset; no dedicated GPIO

// ── I2S port ─────────────────────────────────────────────────────────────────
#define I2S_PORT        I2S_NUM_0

// ── Audio config ─────────────────────────────────────────────────────────────
#define SAMPLE_RATE     16000   // Hz — matches Google STT requirement
#define SAMPLE_BITS     16
#define MAX_RECORD_MS   7000    // max recording length in ms
#define RECORD_BUF_SIZE ((SAMPLE_RATE * SAMPLE_BITS/8 * 2 * MAX_RECORD_MS) / 1000)
// = 16000 * 2 * 2 (stereo) * 7 = 448000 bytes — fits in ESP32-S3's 8 MB PSRAM
// Stereo is required to match ES8311's 32-bit I2S frame. The codec is MONO: one
// stereo slot carries the mic, the other only steady clock/common-mode crosstalk.
// WHICH slot is the mic is NOT assumed anywhere: the recorder energy-picks the
// louder slot per capture, and the wake word picks per poll by speech-likeness
// (see wake_word.h) seeded by the recorder's choice. Mono extraction after
// recording halves the effective byte count.
