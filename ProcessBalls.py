import cv2
import numpy as np
from BallHandler import BallHandler
from util.PlotHsv import plot_hsv
import time

min_ball_radius = 10
max_ball_radius = 100
param1 = 80
param2 = 0.90
dp = 1.8

def process_balls(img: np.ndarray, scale_factor: float = 1, debug: bool = False):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv_frame)
    
    blurred_gray = cv2.GaussianBlur(gray_frame, (13, 13), 0)
    blurred_v = cv2.GaussianBlur(v, (13, 13), 0)
    
    #Apply morphological operations to smooth boundaries between white/colored
    kernel = np.ones((8, 8), np.uint8)
    morph_frame = cv2.morphologyEx(blurred_gray, cv2.MORPH_CLOSE, kernel)
    
    equalized = cv2.equalizeHist(blurred_v)
    cv2.imshow("Equalized", cv2.resize(equalized, (2400, 1400)))
    if debug:
        # cv2.imshow("Morph", cv2.resize(morph_frame, (2400, 1400)))
        cv2.imshow("Blurred V", cv2.resize(blurred_v, (2400, 1400)))

    # Find circles in both processed images
    """   
    circles_gray = cv2.HoughCircles(
        morph_frame,
        cv2.HOUGH_GRADIENT_ALT,
        dp=dp,
        minDist=30,
        param1=param1,
        param2=param2, 
        minRadius=min_ball_radius * scale_factor,
        maxRadius=max_ball_radius * scale_factor
    ) """
    
    circles_v = cv2.HoughCircles(
        blurred_v,
        cv2.HOUGH_GRADIENT_ALT,
        dp=dp,
        minDist=30,
        param1=param1,
        param2=param2,
        minRadius=min_ball_radius * scale_factor,
        maxRadius=max_ball_radius * scale_factor
    )
    
    #get the circles
    all_circles = []
    # if circles_gray is not None:
        # all_circles.extend(circles_gray[0])
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
            circles.append([int(new_circle[0]), int(new_circle[1]), int(new_circle[2])])

    if debug:
        print(f"Found {len(circles)} circles")
        debug_img = img.copy()
        for circle in circles:
            x, y, r = circle
                        
            plot_hsv(debug_img, x, y, 40)
            
            cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
            # cv2.circle(debug_img, (x, y), 2, (0, 0, 255), 3)

            
        debug_img = cv2.resize(debug_img, (2400, 1400))
        cv2.imshow("Balls", debug_img)

    return circles 


def draw_pool_balls(frame: np.ndarray, circles: list[int], ball_handler: BallHandler):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    found_balls = {}
    for ball_name in ball_handler.balls:
        found_balls[ball_name] = False
        
    start = time.time()
    for circle in circles:
        x, y, r = circle
        ball_name = ball_handler.classify_ball_3(x, y, r, hsv)

        if ball_name is not None:
            cv2.putText(frame, (str(int(ball_name[0])) + "," + str(int(ball_name[1])) + "," + str(int(ball_name[2]))), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        """
        ball_name = ball_handler.classify_ball(x, y, r, hsv)
        if ball_name is not None:
            found_balls[ball_name] = True
            ball_handler.update_ball(x, y, r, ball_name)
    """   
    end = time.time()
    print(f"Time taken: {end - start}")

    for ball_name in found_balls:
        if not found_balls[ball_name]:
            ball_handler.not_found(ball_name)

    ball_handler.draw_balls(frame)
    
def process_balls_v2(img: np.ndarray, scale_factor: float = 1, debug: bool = False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Step 1: Apply preprocessing to handle shadows and lighting variations
    # Perform histogram equalization to improve contrast
    
    # Step 2: Use bilateral filter to reduce noise while preserving edges
    
    # Step 3: Detect circles using Hough Circle Transform
    # Convert to grayscale for circle detection
    blur_size = 5 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    edges = cv2.Canny(gray, 50, 150)
    cv2.imshow("Edges", edges)
    # morph_frame = cv2.morphologyEx(morph_frame, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    # Apply Hough Circle Transform
    # Parameters need to be tuned based on your specific setup
    """ 
    circles = cv2.HoughCircles(
        edges, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2,              # Resolution of accumulator
        minDist=30,        # Minimum distance between circles
        param1=65,         # Upper threshold for Canny edge detector
        param2=30,         # Threshold for center detection
        minRadius=5,      # Min radius to be detected
        maxRadius=50       # Max radius to be detected
    )
    """
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT_ALT, 
        dp=1.2,              # Resolution of accumulator
        minDist=30,        # Minimum distance between circles
        param1=40,         # Upper threshold for Canny edge detector
        param2=0.60,         # Threshold for center detection
        minRadius=5,      # Min radius to be detected
        maxRadius=100       # Max radius to be detected
    )
    
    
    detected_balls = []
    
    if circles is not None:
        for circle in circles[0]:
            x, y, r = circle
            x, y, r = int(x), int(y), int(r)
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    
    cv2.imshow("Detected Balls", img) 
    
    
def process_balls_new(img: np.ndarray):
    new_img = img.copy()
    
    h,s,v = cv2.split(cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV))
    
    blur = cv2.GaussianBlur(v, (5, 5), 0)
    
    circles = cv2.HoughCircles(
        blur, 
        cv2.HOUGH_GRADIENT_ALT, 
        dp=1.2,              # Resolution of accumulator
        minDist=30,        # Minimum distance between circles
        param1=40,         # Upper threshold for Canny edge detector
        param2=0.60,         # Threshold for center detection
        minRadius=5,      # Min radius to be detected
        maxRadius=100       # Max radius to be detected
    )
    
    if circles is not None:
        print(len(circles[0]))
        for circle in circles[0]:
            x, y, r = circle
            x, y, r = int(x), int(y), int(r)
            cv2.circle(new_img, (x, y), r, (0, 255, 0), 2)
    

    cv2.imshow("Balls", new_img)