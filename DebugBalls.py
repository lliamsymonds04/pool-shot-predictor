import cv2
import numpy as np

from ProcessTable import process_table, find_balls, remove_table_green, make_balls_white, merge_balls, warp_table
from ProcessBalls import classify_balls, classify_ball, colours, BallClassification
from util.TouchupImage import touchup_image

def debug_table(img: np.ndarray):
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

    #debug the balls
    debug_img = draw_balls_classificiation(raw_table, merged_balls, ball_classifications)
    cv2.imshow("Debug Balls", debug_img)
    
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
   
def debug_classify_ball(balls: list[tuple[int]], table_img: np.ndarray, index: int):
    ball = balls[index]
    x,y,r = ball
    hsv_image = cv2.cvtColor(table_img, cv2.COLOR_BGR2HSV)
    result = classify_ball(x, y, r, hsv_image, True)

    
    debug_img = table_img.copy()
    #draw the ball
    cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
    cv2.circle(debug_img, (x, y), 2, (0, 0, 255), 3) # Draw center of circle
    
    cv2.imshow("Debug", debug_img)
    return result

RING_THICKNESS = 3
WHITE_RING_THICKNESS = 2
RING_OFFSET = 2
def draw_ball(x: int, y: int, radius: int, colour: str, stripped: bool, number: int, img: np.ndarray):
    """
    Draws the ball on the image
    """
    #get the hsv colour of the ball
    hsv = colours[colour]
    #convert to bgr
    b, g, r = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0]
    
    # Convert numpy.uint8 to Python int for OpenCV
    ring_colour = (int(b), int(g), int(r))
    #draw the coloured circle
    radius += RING_OFFSET
    cv2.circle(img, (x, y), radius, ring_colour, RING_THICKNESS)
    cv2.putText(img, str(number), (x + int(radius * 1.5), int(y - radius * 1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if stripped:
        cv2.circle(img, (x, y), radius + RING_THICKNESS, (255, 255, 255), WHITE_RING_THICKNESS)
        
    return img

def draw_balls_classificiation(table_img: np.ndarray, balls: list[tuple[int]], classifications: list[BallClassification]):
    """
    Draws the balls on the image
    """
    new_img = table_img.copy()
    for i, ball in enumerate(balls):
        x, y, r = ball
        c = classifications[i]
        new_img = draw_ball(x, y, r, c.colour, c.stripped, i, new_img)
        
    return new_img

if __name__ == "__main__":
    img_name = "broken_topdown_1.jpg"
    img_path = f"images/table/{img_name}"
    img = cv2.imread(img_path)
    debug_table(img)