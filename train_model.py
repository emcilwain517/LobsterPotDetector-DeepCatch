import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pathlib
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 20
DATASET_DIR = pathlib.Path(__file__).parent / "Dataset_Resized"
MODEL_OUT = pathlib.Path(__file__).parent / "model"

# load dataset splits
def load_split(split_name):
    return keras.utils.image_dataset_from_directory(
        DATASET_DIR / split_name,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary",        # 0 = lobster, 1 = not_lobster alphabtical
        shuffle=(split_name == "train"),
    )

train_ds = load_split("train")
val_ds = load_split("val")
test_ds = load_split("test")
class_names = train_ds.class_names
print(f"Classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# the mobilenetv2 backbone - alpha 0.5 gives a good balance for esp32
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    alpha=0.5,
)
base_model.trainable = False

model = keras.Sequential([
    layers.Rescaling(1.0 / 127.5, offset=-1, input_shape=(IMG_SIZE, IMG_SIZE, 3)),

    # augmentation - vertical flip included since camera looks top down into pot
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.15),

    base_model,

    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# initial training with frozen backbone
print("\n--- Training (frozen backbone) ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
    ],
)

# finetune the last 20 layers with lower lr
print("\n--- Fine-tuning ---")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
    ],
)

print("\n--- Test set evaluation ---")
loss, accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {accuracy:.2%}")

# save keras model
MODEL_OUT.mkdir(exist_ok=True)
keras_path = MODEL_OUT / "lobster_model.keras"
model.save(keras_path)
print(f"Saved model to {keras_path}")

# convert to int8 quantised tflite
print("\nConverting to TFLite (int8 quantised)...")

def representative_dataset():
    for images, _ in train_ds.take(10):
        for i in range(min(10, images.shape[0])):
            yield [tf.expand_dims(images[i], 0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

tflite_path = MODEL_OUT / "lobster_model.tflite"
tflite_path.write_bytes(tflite_model)
print(f"TFLite model saved: {tflite_path} ({len(tflite_model) / 1024:.1f} KB)")

# genreate C header for embedding in firmware
header_path = pathlib.Path(__file__).parent / "firmware" / "lobster_detector" / "src" / "lobster_model.h"

with open(header_path, "w") as f:
    f.write("#ifndef LOBSTER_MODEL_H\n#define LOBSTER_MODEL_H\n\n")
    f.write("const unsigned char lobster_model[] = {\n")
    for i, byte in enumerate(tflite_model):
        if i % 12 == 0:
            f.write("    ")
        f.write(f"0x{byte:02x}")
        if i < len(tflite_model) - 1:
            f.write(", ")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"const unsigned int lobster_model_len = {len(tflite_model)};\n\n")
    f.write("#endif // LOBSTER_MODEL_H\n")

print(f"C header saved: {header_path}")
print("\nDone! Model is ready for ESP32-S3.")