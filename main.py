import cv2
import numpy as np

from ProcessTable import process_table, find_balls, remove_table_green, make_balls_white, merge_balls
from ProcessBalls import draw_balls_debug, classify_balls, debug_classify_ball, draw_balls_classificiation
from BallHandler import BallHandler
from util.TouchupImage import touchup_image

def process_image(img: np.ndarray):
    ball_handler = BallHandler()
    
    touchedup_image = touchup_image(img)
    table, warp_matrix = process_table(touchedup_image, 1000)
    if table is None:
        return
    
    cv2.imshow("Table", table)
    

    removed_green = remove_table_green(table)
    white_balls = make_balls_white(removed_green)
    balls, im1 = find_balls(removed_green)
    balls2, im2 = find_balls(white_balls)
    # cv2.imshow("Balls", im1)
    # cv2.imshow("Balls2", im2)
    merged_balls = merge_balls(balls, balls2)
    # new_balls = draw_balls_debug(removed_green, merged_balls)
    # cv2.imshow("Merged Balls", new_balls)
    # classify_balls(merged_balls, removed_green)
    # print(f"classified ball as {debug_classify_ball(merged_balls, removed_green, 0)}");
    ball_classifications = classify_balls(merged_balls, removed_green)
    print(ball_classifications)
    result = draw_balls_classificiation(table, merged_balls, ball_classifications)
    cv2.imshow("Classified Balls", result)
    

    
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the image
    image_name = "broken_topdown_1"
    image_path = f"images/table/{image_name}.jpg"

    img = cv2.imread(image_path)
    process_image(img)