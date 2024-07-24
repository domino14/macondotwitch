import cv2
import numpy as np
import requests


def detect_corners_via_api(image_path, show=False, roi=[[500, 50], [1400, 800]]):
    # Load the image
    image = cv2.imread(image_path)

    # Define the ROI (center of the image)
    roi_x1, roi_y1 = roi[0]
    roi_x2, roi_y2 = roi[1]
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

    # Save the ROI to a temporary file
    roi_image_path = "roi_image.jpg"
    cv2.imwrite(roi_image_path, roi)

    # Send the image to the API for corner detection
    api_url = "https://scrabblecam.com/process"
    with open(roi_image_path, "rb") as f:
        response = requests.post(api_url, files={"file": f})
        rjson = response.json()
        print(rjson)
    # Check if the request was successful
    if response.status_code != 200:
        print("Failed to detect corners")
        return None

    # Parse the response to get the corners
    corners = rjson.get("corners", [])
    if not corners:
        print("No corners detected")
        return None

    # Convert corners to a NumPy array and offset to match the original image
    in_points = np.array(
        [[x + roi_x1, y + roi_y1] for x, y in corners], dtype=np.float32
    )
    if show:
        disp = image.copy()
        # Draw the detected corners on the image
        for x, y in in_points:
            cv2.circle(disp, (int(x), int(y)), 1, (0, 0, 255), -1)

        # Draw a rectangle around the ROI
        cv2.rectangle(disp, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 255), 2)

        # Display the image with detected corners and ROI
        cv2.imshow("Detected Corners via API", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image, in_points


def refine_corners(image, corners, show=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined_corners = cv2.cornerSubPix(gray, corners, (25, 25), (2, 2), criteria)

    if show:
        disp = image.copy()
        # Draw the detected corners on the image
        for x, y in refined_corners:
            cv2.circle(disp, (int(x), int(y)), 1, (0, 0, 255), -1)

        # Display the image with detected corners and ROI
        cv2.imshow("Refined corners", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return refined_corners


def warp_image_to_square(image, in_points, show=False):
    # Define the output points for the 1000x1000 square
    out_points = np.float32([[50, 50], [1050, 50], [1050, 1050], [50, 1050]])

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(in_points, out_points)

    # Apply the perspective transformation
    img_warped = cv2.warpPerspective(
        image,
        matrix,
        (1100, 1100),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if show:
        # Display the warped image
        cv2.imshow("Warped Image", img_warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_warped


if __name__ == "__main__":
    # Example usage
    image_path = "output_frame.jpg"
    result = detect_corners_via_api(image_path)
    if result is not None:
        image, in_points = result
        warp_image_to_square(image, in_points)
