import cv2
import os

from corners import detect_corners_via_api, refine_corners, warp_image_to_square


def split_and_save_squares(img_warped, save_dir):
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Size of the scrabble board
    board_size = 1000.0
    square_size = board_size / 15.0

    # Split the board into squares and save each square
    for row in range(15):
        for col in range(15):
            x_start = int(round(50 + col * square_size))
            y_start = int(round(50 + row * square_size))
            square = img_warped[
                y_start : y_start + int(round(square_size)),
                x_start : x_start + int(round(square_size)),
            ]
            square_filename = os.path.join(save_dir, f"square_{row}_{col}.png")
            cv2.imwrite(square_filename, square)
            print(f"Saved {square_filename}")


def draw_grid_on_image(img_warped, show=False):
    # Size of the scrabble board
    board_size = 1000.0
    square_size = board_size / 15.0

    # Draw the grid
    for row in range(16):
        y = int(round(50 + row * square_size))
        cv2.line(img_warped, (50, y), (1050, y), (0, 255, 0), 2)
    for col in range(16):
        x = int(round(50 + col * square_size))
        cv2.line(img_warped, (x, 50), (x, 1050), (0, 255, 0), 2)

    # Display the image with the grid
    if show:
        cv2.imshow("Warped Image with Grid", img_warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def create_ref_data(image_path, save_dir):
    # Detect corners via API and warp the image (from previous steps)
    result = detect_corners_via_api(image_path, True, [[0, 0], [3000, 3000]])

    if result is not None:
        image, in_points = result
        # refined_corners = refine_corners(image, in_points, True)
        warped_image = warp_image_to_square(image, in_points, True)
        draw_grid_on_image(warped_image.copy(), True)
        split_and_save_squares(warped_image, save_dir)


if __name__ == "__main__":
    image_path = "hires.png"
    save_dir = "refdata"  # Directory to save the squares
    create_ref_data(image_path, save_dir)
