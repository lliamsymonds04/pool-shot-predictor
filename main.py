import cv2
import numpy as np

from ProcessBalls import process_balls, draw_ball_circles, process_contours
from ProcessTable import process_table, highlight_balls_on_table, convolve_balls, convolve_balls_hough
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
    # r = convolve_balls(result, 3)
    # cv2.imshow("Convolved Balls", r)
    
    # balls = process_contours(r)
    """ 
    if balls is not None:
        new_table = draw_ball_circles(table, balls)
        cv2.imshow("Balls", new_table)
    """
    convolve_balls_hough(result, 5)
    
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the image
    image_name = "broken_topdown_1"
    image_path = f"images/table/{image_name}.jpg"

    img = cv2.imread(image_path)
    process_image(img)