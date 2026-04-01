// Lobster pot detector -  ESP32-S3 WROOM with OV3660
// captures frames, runs TFLite Micro inference and controls LEDs

#include <Arduino.h>
#include "esp_camera.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "lobster_model.h"

// camera pins for the freenove ESP32 cam model
#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   15
#define SIOD_GPIO_NUM    4
#define SIOC_GPIO_NUM    5
#define Y9_GPIO_NUM     16
#define Y8_GPIO_NUM     17
#define Y7_GPIO_NUM     18
#define Y6_GPIO_NUM     12
#define Y5_GPIO_NUM     10
#define Y4_GPIO_NUM      8
#define Y3_GPIO_NUM      9
#define Y2_GPIO_NUM     11
#define VSYNC_GPIO_NUM   6
#define HREF_GPIO_NUM    7
#define PCLK_GPIO_NUM   13

#define WHITE_LED_PIN  1   // capture/detection indicator
#define BLUE_LED_PIN   2   // lobster detected

#define IMG_SIZE       160
#define DETECTION_THRESHOLD  0.3
#define CAPTURE_INTERVAL_MS  3000

// tensor arena in PSRAM - 1MB needed for 160x160 input
constexpr int kTensorArenaSize = 1024 * 1024;
uint8_t *tensor_arena = nullptr;

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input_tensor = nullptr;
TfLiteTensor *output_tensor = nullptr;

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;  // 20MHz for freenove board
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size   = FRAMESIZE_240X240;  // square capture, downscaled to 160x160
  config.fb_count     = 1;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  // ov3660 specific tweaks
  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }

  Serial.println("Camera initialised (OV3660, 240x240 -> 160x160 RGB565)");
  return true;
}

bool initModel() {
  model = tflite::GetModel(lobster_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema mismatch: got %lu, expected %d\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  tensor_arena = (uint8_t *)ps_malloc(kTensorArenaSize);
  if (!tensor_arena) {
    Serial.println("Failed to allocate tensor arena in PSRAM");
    return false;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return false;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.printf("Model loaded. Input: [%d, %d, %d, %d] type=%d\n",
                input_tensor->dims->data[0], input_tensor->dims->data[1],
                input_tensor->dims->data[2], input_tensor->dims->data[3],
                input_tensor->type);
  Serial.printf("Tensor arena used: %zu / %d bytes\n",
                interpreter->arena_used_bytes(), kTensorArenaSize);

  return true;
}

// downscale 240x240 RGB565 frame to 160x160 uint8 RGB for the model
void preprocessFrame(camera_fb_t *fb) {
  uint8_t *input_data = input_tensor->data.uint8;
  uint16_t *rgb565 = (uint16_t *)fb->buf;
  int src_size = fb->width;

  for (int y = 0; y < IMG_SIZE; y++) {
    for (int x = 0; x < IMG_SIZE; x++) {
      int src_x = (x * src_size) / IMG_SIZE;
      int src_y = (y * src_size) / IMG_SIZE;
      uint16_t raw = rgb565[src_y * src_size + src_x];
      uint16_t pixel = (raw >> 8) | (raw << 8);

      // extract RGB channels from RGB565 format
      input_data[(y * IMG_SIZE + x) * 3 + 0] = (pixel >> 8) & 0xF8;
      input_data[(y * IMG_SIZE + x) * 3 + 1] = (pixel >> 3) & 0xFC;
      input_data[(y * IMG_SIZE + x) * 3 + 2] = (pixel << 3) & 0xF8;
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== Lobster Pot Detector ===\n");

  pinMode(WHITE_LED_PIN, OUTPUT);
  pinMode(BLUE_LED_PIN, OUTPUT);
  digitalWrite(WHITE_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW);

  // flash both LEDs on startup as a quick test
  digitalWrite(WHITE_LED_PIN, HIGH);
  digitalWrite(BLUE_LED_PIN, HIGH);
  delay(500);
  digitalWrite(WHITE_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW);

  if (!initCamera()) {
    Serial.println("FATAL: Camera init failed. Halting.");
    while (true) { delay(1000); }
  }

  if (!initModel()) {
    Serial.println("FATAL: Model init failed. Halting.");
    while (true) { delay(1000); }
  }

  Serial.println("\nReady. Starting detection loop...\n");
}

void loop() {
  digitalWrite(WHITE_LED_PIN, HIGH);

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Frame capture failed");
    digitalWrite(WHITE_LED_PIN, LOW);
    delay(1000);
    return;
  }

  preprocessFrame(fb);
  esp_camera_fb_return(fb);

  unsigned long t0 = millis();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    digitalWrite(WHITE_LED_PIN, LOW);
    delay(1000);
    return;
  }
  unsigned long inference_ms = millis() - t0;

  // model output is uint8 quantised sigmoid - 0=lobster, 255=not_lobster
  uint8_t raw_output = output_tensor->data.uint8[0];
  float confidence = raw_output / 255.0f;
  float lobster_conf = (1.0f - confidence) * 100.0f;
  float not_lobster_conf = confidence * 100.0f;
  bool lobster_detected = (confidence < DETECTION_THRESHOLD);

  // dead zone if neither class is confident (both below 70%) treat as uncertain
  bool uncertain = (lobster_conf < 70.0f && not_lobster_conf < 70.0f);
  if (uncertain) {
    lobster_detected = false;
  }

  digitalWrite(BLUE_LED_PIN, lobster_detected ? HIGH : LOW);
  digitalWrite(WHITE_LED_PIN, LOW);

  if (uncertain) {
    Serial.printf("    Uncertain.       (lobster: %.1f%%, not: %.1f%%, raw: %d, %lums)\n",
                  lobster_conf, not_lobster_conf, raw_output, inference_ms);
  } else if (lobster_detected) {
    Serial.printf(">>> LOBSTER DETECTED! (conf: %.1f%%, raw: %d, %lums)\n",
                  lobster_conf, raw_output, inference_ms);
  } else {
    Serial.printf("    No lobster.      (conf: %.1f%%, raw: %d, %lums)\n",
                  not_lobster_conf, raw_output, inference_ms);
  }

  delay(CAPTURE_INTERVAL_MS);
}