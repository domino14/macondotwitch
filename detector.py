import cv2
import os
from corners import detect_corners_via_api, warp_image_to_square, refine_corners
from skimage.metrics import structural_similarity as ssim


def split_board_into_squares(img_warped):
    # Define the start point and size of the actual board within the warped image
    start_x = 50
    start_y = 50
    board_size = 1000
    square_size = board_size / 15

    squares = []
    for row in range(15):
        for col in range(15):
            x_start = int(round(start_x + col * square_size))
            y_start = int(round(start_y + row * square_size))
            square = img_warped[
                y_start : y_start + int(round(square_size)),
                x_start : x_start + int(round(square_size)),
            ]
            squares.append(square)

    return squares


def template_matching(square, templates):
    best_match = None
    max_val = -1

    for tile, template in templates.items():
        res = cv2.matchTemplate(square, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val_tmp, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val_tmp > max_val:
            max_val = max_val_tmp
            best_match = tile

    return best_match, max_val


def load_templates(template_dir):
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith(".png"):
            tile = filename[0]
            template_path = os.path.join(template_dir, filename)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            templates[tile] = template
    return templates


def detect_tiles_on_board(img_warped, templates):
    squares = split_board_into_squares(img_warped)
    board = []

    for i, square in enumerate(squares):
        gray_square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
        tile, confidence = template_matching(gray_square, templates)
        board.append((tile, confidence, gray_square))

    return board


def visualize_squares(squares):
    for i, square in enumerate(squares):
        cv2.imshow(f"Square {i}", square)
        cv2.waitKey(0)  # Wait for a key press to move to the next square

    cv2.destroyAllWindows()


def visualize_comparisons(square, comparison_images):
    for template, similarity_index in comparison_images:
        combined_image = cv2.hconcat([square, template])
        combined_image = cv2.resize(
            combined_image, (600, 300), interpolation=cv2.INTER_LINEAR
        )
        cv2.putText(
            combined_image,
            f"Similarity: {similarity_index:.2f}",
            (10, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Square Comparison", combined_image)
        cv2.waitKey(0)  # Wait for a key press to move to the next comparison

    cv2.destroyAllWindows()


def detect_board(image_path, template_dir):
    # Detect corners via API and warp the image (from previous steps)
    result = detect_corners_via_api(image_path)

    if result is not None:
        image, in_points = result
        refined_corners = refine_corners(image, in_points, True)

        if abs(refined_corners[0][1] - refined_corners[1][1]) > 60:
            # too tilted
            return

        warped_image = warp_image_to_square(image, refined_corners)

        # Load tile templates
        templates = load_templates(template_dir)

        # Visualize the squares split from the warped board
        squares = split_board_into_squares(warped_image)

        # Detect tiles on the board
        board = detect_tiles_on_board(warped_image, templates)

        # # Visualize comparisons for row 0, column 7
        target_index = 55
        tile, confidence, square = board[target_index]
        # visualize_comparisons(square, comparison_images)

        # # Print the board detection results
        for row in range(15):
            for col in range(15):
                tile, confidence, _ = board[row * 15 + col]
                print(f"Square ({row}, {col}): {tile} (Confidence: {confidence:.2f})")

        # Optionally: Display the warped image
        cv2.imshow("Warped Image", warped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "tilted.png"
    template_dir = "refdata"  # Directory containing template images of Scrabble tiles
    detect_board(image_path, template_dir)
