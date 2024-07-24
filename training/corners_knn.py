import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load corner data
with open("corner_data.json", "r") as f:
    corner_data = json.load(f)

# Prepare data for training
X = []
y = []
for img_path, corners in corner_data.items():
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten()
    X.append(gray)
    y.append(np.array(corners).flatten())

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a KNN regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate the model
y_pred = knn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model
joblib.dump(knn, "corner_detection_model.pkl")

# Example usage of the trained model
# def predict_corners(image_path):
#     model = joblib.load('corner_detection_model.pkl')
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten().reshape(1, -1)
#     corners = model.predict(gray).reshape(4, 2)
#     return corners

# # Predict corners for a new image
# image_path = "new_image.jpg"
# predicted_corners = predict_corners(image_path)
# print(f"Predicted corners: {predicted_corners}")
