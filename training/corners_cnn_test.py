import cv2
import numpy as np
import tensorflow as tf


def predict_corners(
    image_path, model_path="corner_detection_cnn.h5", target_size=(512, 512)
):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the input image
    image = cv2.imread(image_path)
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized.astype("float32") / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)

    # Predict corners
    corners = model.predict(image_input).reshape(4, 2)

    # Scale corners back to the original image size
    scale_x, scale_y = (
        original_size[1] / target_size[1],
        original_size[0] / target_size[0],
    )
    corners = corners * [scale_x, scale_y]

    # Draw the predicted corners on the original image
    for corner in corners:
        cv2.circle(image, tuple(corner.astype(int)), 5, (0, 0, 255), -1)

    # Display the image with the drawn corners
    cv2.imshow("Predicted Corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return corners


# Example usage
predict_corners("../tilted.png")
