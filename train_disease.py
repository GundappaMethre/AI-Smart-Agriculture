
import os
import json
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Dataset location
DATA_DIR = "data/plantvillage_download/raw/color"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

# Find tomato disease folders only
class_names = sorted([
    folder for folder in os.listdir(DATA_DIR)
    if folder.startswith("Tomato___")
    and os.path.isdir(os.path.join(DATA_DIR, folder))
])

if len(class_names) < 2:
    raise ValueError(
        "Tomato dataset folders not found. "
        "Wait for PlantVillage download to finish."
    )

print("Tomato classes:", class_names)

# Load training images
train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    class_names=class_names,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation images
val_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    class_names=class_names,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Preprocess images
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.map(
    lambda x, y: (preprocess_input(x), y)
).prefetch(AUTOTUNE)

val_data = val_data.map(
    lambda x, y: (preprocess_input(x), y)
).prefetch(AUTOTUNE)

# Load pretrained MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# Build classification model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
print("Training tomato disease detection model...")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save model
model.save(
    "models/crop_disease_model.keras"
)

# Save disease class names
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("Tomato disease model trained successfully!")
print("Model saved in models/crop_disease_model.keras")

