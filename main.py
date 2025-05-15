import cv2
import numpy as np

from ProcessTable import process_table, find_balls, remove_table_green, make_balls_white
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
    cv2.imshow("Removed Green", removed_green)
    white_balls = make_balls_white(removed_green)
    cv2.imshow("White Balls", white_balls)
    balls = find_balls(removed_green)
    
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the image
    image_name = "broken_topdown_1"
    image_path = f"images/table/{image_name}.jpg"

    img = cv2.imread(image_path)
    process_image(img)