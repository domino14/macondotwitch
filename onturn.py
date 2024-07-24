import cv2
import numpy as np


def detect_player_on_turn(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Define the regions of interest (ROIs) for the timers
    roi1_coords = (650, 950, 950, 1050)  # x1, y1, x2, y2
    roi2_coords = (960, 950, 1260, 1050)  # (x1, y1, x2, y2)
    # Extract the ROIs
    roi1 = image[roi1_coords[1] : roi1_coords[3], roi1_coords[0] : roi1_coords[2]]
    roi2 = image[roi2_coords[1] : roi2_coords[3], roi2_coords[0] : roi2_coords[2]]

    # Convert the entire image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the ROIs to grayscale
    gray_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    # Calculate the mean intensity of the ROIs
    mean_intensity1 = np.mean(gray_roi1)
    mean_intensity2 = np.mean(gray_roi2)

    # Define a threshold to detect the grey background
    grey_threshold = 60  # Adjust this threshold based on the actual grey level

    # Draw rectangles around the ROIs for visualization on the grayscale image
    cv2.rectangle(
        gray_image,
        (roi1_coords[0], roi1_coords[1]),
        (roi1_coords[2], roi1_coords[3]),
        (255, 0, 0),
        2,
    )
    cv2.rectangle(
        gray_image,
        (roi2_coords[0], roi2_coords[1]),
        (roi2_coords[2], roi2_coords[3]),
        (255, 0, 0),
        2,
    )

    # Display the grayscale image with rectangles
    cv2.imshow("Grayscale ROIs", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Determine which ROI has the grey background
    if mean_intensity1 > grey_threshold and mean_intensity2 < grey_threshold:
        return "Player 1 is on turn"
    elif mean_intensity2 > grey_threshold and mean_intensity1 < grey_threshold:
        return "Player 2 is on turn"
    else:
        return "Cannot determine the player on turn"


# Example usage
image_path = "output_frame.jpg"
player_on_turn = detect_player_on_turn(image_path)
print(player_on_turn)
