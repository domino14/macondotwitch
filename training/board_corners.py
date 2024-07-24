import cv2
import numpy as np
import glob
import json


def refine_corner(image, point):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = np.array([[point]], dtype=np.float32)
    refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined_corners[0][0]


def click_event(event, x, y, flags, params):
    global corners, corner_index, image_display
    if event == cv2.EVENT_LBUTTONDOWN:
        refined_corner = refine_corner(params["image"], (x, y))
        corners[corner_index] = refined_corner

        # Show the refined corner on a copy of the image
        image_display = params["image"].copy()
        for i in range(corner_index + 1):
            cv2.circle(image_display, tuple(corners[i].astype(int)), 3, (0, 0, 255), -1)
        cv2.imshow("Image", image_display)

        corner_index += 1
        if corner_index == 4:
            corner_index = 0  # Reset for potential redoing


def annotate_corners():
    global corners, corner_index, image_display
    import random

    images = glob.glob("boardcap/*.jpg")
    random.shuffle(images)
    images = images[:100]
    corner_data = {}

    for img_path in images:
        image = cv2.imread(img_path)
        image_display = image.copy()
        corners = np.zeros((4, 2), dtype=np.float32)
        corner_index = 0

        cv2.imshow("Image", image_display)
        cv2.setMouseCallback("Image", click_event, {"image": image})

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):  # Press 'r' to redo the current set of corners
                corner_index = 0
                corners = np.zeros((4, 2), dtype=np.float32)
                image_display = image.copy()
                cv2.imshow("Image", image_display)
            elif key == ord(
                "n"
            ):  # Press 'n' to confirm the corners and move to the next image
                if np.all(corners != 0):  # Ensure all corners are set
                    corner_data[img_path] = corners.tolist()
                    break
            elif key == ord("s"):  # Press 's' to skip the image
                break

        cv2.destroyAllWindows()

    with open("corner_data.json", "w") as f:
        json.dump(corner_data, f)


if __name__ == "__main__":
    annotate_corners()
