import cv2
import joblib


import cv2
import joblib
import numpy as np


def predict_corners(image_path):
    # Load the trained model
    model = joblib.load("corner_detection_model.pkl")

    # Load the input image
    image = cv2.imread(image_path)

    # Resize the image to match the dimensions used during training (e.g., 1440x1440)
    target_size = (1920, 1080)
    resized_image = cv2.resize(image, target_size)

    # Convert the image to grayscale and flatten it
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY).flatten().reshape(1, -1)

    # Predict the corners using the trained model
    corners = model.predict(gray).reshape(4, 2)

    # Draw the predicted corners on the original image (with original dimensions)
    for corner in corners:
        cv2.circle(
            image,
            tuple(
                (
                    corner
                    * (image.shape[1] / target_size[0], image.shape[0] / target_size[1])
                ).astype(int)
            ),
            5,
            (0, 0, 255),
            -1,
        )

    # Display the image with the drawn corners
    cv2.imshow("Predicted Corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return corners


# Example usage
predict_corners("../frame_0005.jpg")
