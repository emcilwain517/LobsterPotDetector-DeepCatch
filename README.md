# DeepCatch - Lobster Pot Monitoring system

A proof of concept embedded image classification system built on the ESP32-S3. The device sits inside a lobster pot and uses a camera and a TFLite machine learning model to detect when a lobster is present, triggering an LED indicator.


# Demo

<img src="IMG_8883-ezgif.com-optimize(1).gif" width="45%"> <img src="ezgif.com-optimize.gif" width="45%">

Full quality video: [YouTube](https://youtu.be/ZmNJcU3zN0Q)

# How it works

The ESP captures a 240x240 image from an OV3660 camera every 3 seconds, downscales it to 160x160 and runs it through a quantised MobileNetV2 model. If a lobster is detected the blue LED turns on. A white LED flashes during each capture cycle.

The model was trained on a dataset of lobster and non-lobster images including AI-generated images of lobsters inside pots to better match the real world camera view. It was converted to int8 TFLite format for deployment on microcontroller.


# Files

- `firmware/lobster_detector/src/main.cpp` - main ESP32 firmware, handles camera capture, preprocessing and inference
- `firmware/lobster_detector/platformio.ini` - PlatformIO build config for the ESP32
- `train_model.py` - training script used to build and export the model
- `model/lobster_model.tflite` - the quantised int8 TFLite model deployed on the device

# Hardware

- Freenove ESP32-S3 WROOM (N16R8 - 16MB flash, 8MB PSRAM)
- OV3660 camera module
- White LED (GPIO 1) and Blue LED (GPIO 2)
- 330 Ohmn Resistors
- BreadBoard


# Notes

The OV3660 auto exposure takes around 10 seconds to adjust when there is a dramatic change in scene brightness. This is normal sensor behaviour and would not be an issue in a real deployment where lighting conditions are relatively stable.