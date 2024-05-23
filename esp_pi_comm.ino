#include "esp_camera.h"
#include <Arduino.h>

// Camera module pin definitions

#define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13
#define BUTTON_PIN        D8
int i = 0;
unsigned long lastPressTime = 0;  // Store the time of the last button press
const unsigned long debounceDelay = 5000;  // 5 seconds debounce delay


void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  pinMode(BUTTON_PIN, INPUT_PULLDOWN);
  //Serial.println("hola");
  //Serial.setDebugOutput(true);
  //Serial.println();
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG; // for streaming
  //config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10;
  config.fb_count = 1;

  if(config.pixel_format == PIXFORMAT_JPEG){
    if(psramFound()){
      config.jpeg_quality = 2;
      config.fb_count = 2;
      config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
      // Limit the frame size when PSRAM is not available
      config.frame_size = FRAMESIZE_HD;
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    // Best option for face detection/recognition
    config.frame_size = FRAMESIZE_HD;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

  // Initialize the camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    //s->set_vflip(s, 1); // Flip it back
    s->set_brightness(s, 10); // Up the brightness just a bit
    //s->set_saturation(s, -2); // Lower the saturation
  }
  
  // Set exposure time (adjust this value as needed)
  s->set_exposure_ctrl(s, 1); // Enable exposure control
  s->set_aec2(s, 0); // Turn off automatic exposure adjustment
  s->set_agc_gain(s, 0); // Turn off automatic gain control
  s->set_aec_value(s, 2000);

  // Drop down frame size for higher initial frame rate
  if(config.pixel_format == PIXFORMAT_JPEG) {
    s->set_framesize(s, FRAMESIZE_QVGA);
  }

#if defined(CAMERA_MODEL_XIAO_ESP32S3)
  s->set_brightness(s, 10);
  //s->set_vflip(s, 1);
  //s->set_hmirror(s, 1);
#endif

}
void loop() {
  i+=1;
   
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    
    if (command == "TRIGGER") {
        digitalWrite(LED_BUILTIN, LOW);
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
          Serial.println("Camera capture failed");
          return;
        }
        while(Serial.available() > 0){
          Serial.read();
        }
        // Send the frame data over Serial
        Serial.write((uint8_t*)&fb->len, sizeof(fb->len)); // Send the length of the image
        Serial.write(fb->buf, fb->len); // Send the image buffer
        //Serial.println("hola");
        // Return the frame buffer back to the driver for reuse
        esp_camera_fb_return(fb);
        // Delay for a bit to avoid flooding the serial port
        
        delay(200);
        digitalWrite(LED_BUILTIN, HIGH);
        
    }    
        
  }
  String take_photo = "Take_Photo";
  unsigned long currentTime = millis();
  if ((digitalRead(BUTTON_PIN) == HIGH) && Serial.available() == 0 && (currentTime - lastPressTime > debounceDelay)){
    Serial.println(take_photo);
    lastPressTime = currentTime;

    //Serial.println((analogRead(BUTTON_PIN)));
  }    
  //Serial.println("hola");
}
