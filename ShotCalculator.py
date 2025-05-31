import cv2
import numpy as np

from MyTypes import BallClassification
from ProcessBalls import colours



def calculate_best_shot(table: np.ndarray, balls: list[tuple[int]], ball_classifications: list[BallClassification]):

    # For now, just print the positions and classifications
    print("Calculating best shot...")
    for ball, classification in zip(balls, ball_classifications):
        print(classification.colour)
        x, y, radius = ball
        hsv = colours[classification.colour]
        b, g, r = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        #convert to int
        b, g, r = int(b), int(g), int(r)
        
        cv2.circle(table, (x, y), radius, (b,g,r), 2)

    cv2.imshow("Best Shot", table)
    
    #draw all the balls on the table