import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


# Function to compute the intersection of two lines given in Hesse normal form
def compute_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    if np.linalg.det(A) == 0:  # Lines are parallel
        return None
    x0, y0 = np.linalg.solve(A, b)
    return [int(np.round(x0)), int(np.round(y0))]


# Function to check if two lines are perpendicular
def are_perpendicular(theta1, theta2):
    angle_diff = abs(theta1 - theta2)
    return np.isclose(angle_diff, np.pi / 2, atol=np.pi / 36)  # 5 degree tolerance


# Load the image
image_path = "tilted.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for the unique blue color in HSV
lower_blue = np.array([90, 120, 120])
upper_blue = np.array([120, 240, 230])

# Create a mask for the blue color
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Define the region of interest (ROI)
roi_x_start, roi_x_end = 329, 1014  # Example values, adjust as needed
roi_y_start, roi_y_end = 6, 560  # Example values, adjust as needed

# Apply the ROI to the mask
blue_mask_roi = blue_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

# Invert the blue mask ROI to highlight the black square
inverted_blue_mask_roi = cv2.bitwise_not(blue_mask_roi)

# Use Canny edge detector to find edges
edges = cv2.Canny(inverted_blue_mask_roi, 50, 150)

# Display the edges
plt.figure(figsize=(10, 10))
plt.imshow(edges, cmap="gray")
plt.title("Edges Detected")
plt.show()

# Use Hough Line Transform to detect lines in the edge image
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

# Draw the lines on the image
lines_image = np.copy(image_rgb)

if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(
            lines_image,
            (x1 + roi_x_start, y1 + roi_y_start),
            (x2 + roi_x_start, y2 + roi_y_start),
            (0, 255, 0),
            2,
        )

# Display the image with lines detected
plt.figure(figsize=(10, 10))
plt.imshow(lines_image)
plt.title("Lines Detected")
plt.show()

# Find intersections of the lines to form corners
if lines is not None:
    intersections = []
    perpendicular_pairs = []
    for line1, line2 in combinations(lines, 2):
        if are_perpendicular(line1[0][1], line2[0][1]):
            intersection = compute_intersection(line1[0], line2[0])
            if intersection is not None:
                intersections.append(intersection)
                perpendicular_pairs.append((line1[0], line2[0]))

    # Filter intersections within the ROI bounds
    intersections = [
        pt
        for pt in intersections
        if 0 <= pt[0] < edges.shape[1] and 0 <= pt[1] < edges.shape[0]
    ]

    # Draw the intersection points
    intersections_image = np.copy(lines_image)
    for pt in intersections:
        pt[0] += roi_x_start
        pt[1] += roi_y_start
        cv2.circle(intersections_image, tuple(pt), 5, (255, 0, 0), -1)

    # Display the image with intersection points
    plt.figure(figsize=(10, 10))
    plt.imshow(intersections_image)
    plt.title("Intersections Detected")
    plt.show()

    print("how many intersectiosns", len(intersections), intersections)

    # Find the smallest square that is still large enough
    min_area = 100000  # Minimum area in pixels
    best_square = None
    for corners in combinations(intersections, 4):
        corners = np.array(corners)
        rect = cv2.minAreaRect(corners)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)
        if area >= min_area and (
            best_square is None or area < cv2.contourArea(best_square)
        ):
            print("Best square", corners)
            best_square = box

    # Draw the best square on the original image
    if best_square is not None:
        # Draw the intermediate box for debugging
        debug_image = np.copy(image_rgb)
        for point in best_square:
            cv2.circle(
                debug_image,
                (point[0] + roi_x_start, point[1] + roi_y_start),
                5,
                (0, 255, 255),
                -1,
            )
        cv2.drawContours(debug_image, [best_square], -1, (255, 255, 0), 2)
        plt.figure(figsize=(10, 10))
        plt.imshow(debug_image)
        plt.title("Intermediate Box for Debugging")
        plt.show()

        # Adjust the coordinates based on the ROI offset
        best_square[:, 0] += roi_x_start
        best_square[:, 1] += roi_y_start

        cv2.drawContours(image_rgb, [best_square], -1, (255, 0, 0), 3)
        for point in best_square:
            cv2.circle(image_rgb, tuple(point), 5, (0, 255, 0), -1)  # Draw corners

        # Display the original image with the detected square shape
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.title("Detected Smallest Square Shape with Corners")
        plt.show()

        # Print the coordinates of the corners
        print("Coordinates of the corners:")
        for point in best_square:
            print(tuple(point))
    else:
        print("No square detected.")
else:
    print("No lines detected.")
