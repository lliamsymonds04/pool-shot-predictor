import cv2
import numpy as np

def process_balls(img: np.ndarray, blur_size: int = 5):
    new_img = img.copy()
    
    _,_,v = cv2.split(cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV))
    
    blur = cv2.GaussianBlur(v, (blur_size, blur_size), 0)
    
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
        output = []
        for circle in circles[0]:
            x, y, r = circle
            output.append((int(x), int(y), int(r)))
            
        return output

    return None

def draw_ball_circles(table_img: np.ndarray, balls: tuple[tuple[int]]):
    for ball in balls:
        x, y = ball
        cv2.circle(table_img, (x, y), 20, (0, 255, 0), 2)
        cv2.circle(table_img, (x, y), 2, (0, 0, 255), 3) # Draw center of circle
        
    return table_img

def process_contours(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold or find edges to get contours
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the image to draw on
    ring_image = img.copy()


    ball_origins = []
    for contour in contours:
        # Get the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Ring thickness
        thickness = 2  # pixels
        
        if radius > 35:
            continue

        # Draw outer circle (larger radius)
        cv2.circle(ring_image, center, radius + thickness, (0, 255, 0), thickness)  # Filled
        ball_origins.append((int(x), int(y)))
        
    # cv2.imshow("Ringed Image", ring_image)
    
    return ball_origins