import cv2
import numpy as np
import time

from ProcessBalls import process_balls, draw_pool_balls, process_balls_v2, process_balls_new
from ProcessTable import process_table, highlight_balls_on_table
from BallHandler import BallHandler
from util.TouchupImage import touchup_image

def process_image(img: np.ndarray):
    ball_handler = BallHandler()
    
    touchedup_image = touchup_image(img)
    table, warp_matrix = process_table(touchedup_image, 1000)
    if table is None:
        return
    
    cv2.imshow("Table", table)
    
    result = highlight_balls_on_table(table)
    cv2.imshow("Highlighted Balls", result)
    
    process_balls_new(result)
    
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the image
    image_name = "broken_topdown_3"
    image_path = f"images/table/{image_name}.jpg"

    img = cv2.imread(image_path)
    process_image(img)