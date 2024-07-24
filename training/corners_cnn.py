import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import glob
import json


# Load and preprocess the dataset
def load_data(image_folder, annotation_file, target_size=(512, 512)):
    with open(annotation_file, "r") as f:
        corner_data = json.load(f)

    X = []
    y = []
    for img_path, corners in corner_data.items():
        image = cv2.imread(img_path)
        original_size = image.shape[:2]

        # Resize image to target size
        image_resized = cv2.resize(image, target_size)
        X.append(image_resized)

        # Scale corner coordinates to be relative to the resized image
        scale_x, scale_y = (
            target_size[1] / original_size[1],
            target_size[0] / original_size[0],
        )
        corners_resized = np.array(corners) * [scale_x, scale_y]
        y.append(corners_resized.flatten())

    X = np.array(X, dtype="float32") / 255.0  # Normalize pixel values
    y = np.array(y, dtype="float32")

    return X, y


# Build the CNN model
def build_model(input_shape=(512, 512, 3)):
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(8, activation="linear"),  # 8 values (4 corners * 2 coordinates each)
        ]
    )

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model


# Load data
image_folder = "boardcap"
annotation_file = "corner_data.json"
X, y = load_data(image_folder, annotation_file, target_size=(512, 512))

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = build_model(input_shape=(512, 512, 3))
model.summary()
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Save the trained model
model.save("corner_detection_cnn.h5")
