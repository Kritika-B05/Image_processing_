import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os


# ===============================
# Paths
# ===============================
train_dir = "dataset/train"
test_dir = "dataset/test"

# ===============================
# Image Generator
# ===============================
datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

# ===============================
# CNN Model
# ===============================
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

# ===============================
# Compile
# ===============================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# Train
# ===============================
model.fit(
    train_data,
    epochs=5,
    validation_data=test_data
)

# ===============================
# Save Model
# ===============================
os.makedirs("model", exist_ok=True)
model.save("model/cnn_model.h5")

print("✅ Model saved successfully at model/cnn_model.h5")