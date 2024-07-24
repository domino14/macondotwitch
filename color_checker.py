import cv2
import numpy as np

# Global variables to store the BGR and HSV values
bgr_value = (0, 0, 0)
hsv_value = (0, 0, 0)


# Function to display BGR and HSV values
def show_values(event, x, y, flags, param):
    global bgr_value, hsv_value
    if event == cv2.EVENT_MOUSEMOVE:
        bgr_value = image[y, x]
        hsv_value = hsv_image[y, x]


# Load the image
image_path = "output_frame.jpg"
image = cv2.imread(image_path)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a window and set a mouse callback function
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", show_values)

while True:
    temp_image = image.copy()
    b, g, r = bgr_value
    h, s, v = hsv_value
    text = f"BGR: ({b}, {g}, {r})  HSV: ({h}, {s}, {v})"
    cv2.putText(
        temp_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.imshow("Image", temp_image)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cv2.destroyAllWindows()
