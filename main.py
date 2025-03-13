import cv2
import numpy as np
from BallHandler import BallHandler
from util.TouchupImage import touchup_image
from util.PlotHsv import plot_hsv

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

ball_handler = BallHandler()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
   
    touched_up = touchup_image(frame)
    
    #use gray space and brightness to find the balls
    gray_frame = cv2.cvtColor(touched_up, cv2.COLOR_BGR2GRAY)
    hsv_frame = cv2.cvtColor(touched_up, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_frame)
    
    blurred_gray = cv2.GaussianBlur(gray_frame, (13, 13), 0)
    blurred_v = cv2.GaussianBlur(v, (13, 13), 0)
    
    #Apply morphological operations to smooth boundaries between white/colored
    kernel = np.ones((5, 5), np.uint8)
    morph_frame = cv2.morphologyEx(blurred_gray, cv2.MORPH_CLOSE, kernel)

    
    # Find circles in both processed images
    circles_gray = cv2.HoughCircles(
        morph_frame,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.5,
        minDist=30,
        param1=100,
        param2=0.95, 
        minRadius=20,
        maxRadius=40
    )
    
    circles_v = cv2.HoughCircles(
        blurred_v,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.5,
        minDist=30,
        param1=100,
        param2=0.95,
        minRadius=10,
        maxRadius=40
    )
    
    #get the circles
    all_circles = []
    if circles_gray is not None:
        all_circles.extend(circles_gray[0])
    if circles_v is not None:
        all_circles.extend(circles_v[0])
        
    circles = []
    for new_circle in all_circles:
        valid = True

        #check the new circle against the old circles
        for circle in circles:
            x1, y1 = new_circle[:2]
            x2, y2 = circle[:2]
            
            #check distances to remove overlapping circles
            if abs(x1 - x2) + abs(y1 - y2) < 30:
                valid = False
                break 
            
        if valid:
            circles.append(new_circle)
            
    found_balls = {}
    for ball in ball_handler.balls.keys():
        found_balls[ball] = False    
    
        
    for circle in circles:
        x, y, r = circle
        
        colour = ball_handler.eval_circle(x, y, r, hsv_frame)
        plot_hsv(frame, x, y, 50)               
        if colour and found_balls.get(colour, False) == False:
            found_balls[colour] = True
            ball_handler.update_ball(x, y, r, colour)
                
    for name, found in found_balls.items():
        if not found:
            ball_handler.not_found(name)

    ball_handler.draw_balls(frame)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()