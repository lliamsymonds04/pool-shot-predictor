import cv2
import numpy as np

from ProcessTable import process_table, find_balls, remove_table_green, make_balls_white, merge_balls, warp_table
from ProcessBalls import classify_balls
from ShotCalculator import calculate_best_shot
from util.TouchupImage import touchup_image

def process_image(img: np.ndarray, stripped: bool = False):
    touchedup_image = touchup_image(img)
    warp_matrix = process_table(touchedup_image, 1000)
    if warp_matrix is None:
        return

    table = warp_table(touchedup_image, warp_matrix, 1000)
    if table is None:
        return
    
    removed_green = remove_table_green(table)
    white_balls = make_balls_white(removed_green)
    balls, im1 = find_balls(removed_green)
    balls2, im2 = find_balls(white_balls)

    merged_balls = merge_balls(balls, balls2)

    ball_classifications = classify_balls(merged_balls, removed_green)
    raw_table = warp_table(img, warp_matrix, 1000)

    calculate_best_shot(raw_table, merged_balls, ball_classifications, stripped)
    
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

def prompt_user_for_image():
    while True:
        image_path = input("Enter the image name: ")

        # attempt to load the image
        img = cv2.imread(image_path)
        if img is not None:
            #split the image name to get the name without extension
            path_folders = image_path.split('/')
            image_name = path_folders[-1]
            print(f"Image '{image_name}' loaded successfully.")
            return image_path

def prompt_user_for_strpped():
    while True:
        stripped_input = input("Are you playing stripped balls? (Y/N): ").strip().lower()
        if stripped_input in ['y', 'n']:
            return stripped_input == 'y'
        print("Invalid input. Please enter 'Y' or 'N'.")

if __name__ == "__main__":
    # Load the image
    # image_name = "broken_topdown_2"
    # image_path = f"images/table/{image_name}.jpg"
    # stripped = False
    image_path = prompt_user_for_image()
    stripped = prompt_user_for_strpped()

    img = cv2.imread(image_path)
    process_image(img, stripped)