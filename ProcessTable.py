import numpy as np
import cv2

lower_green = np.array([50, 40, 0])
upper_green = np.array([95, 120, 255])

def process_table(img: np.ndarray, table_length: int = 800):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #use a mask to find the contours
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No table found")
        return 

    #find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour, True)
    perimeter = cv2.arcLength(hull, True)

    #adjust the epsilon factor to get a 4 sided polygon
    epsilon_factor = 0.02
    for _ in range(10):
        approx = cv2.approxPolyDP(hull, epsilon_factor * perimeter, True)
        if len(approx) == 4:
            break     
        
        epsilon_factor += 0.002

    if len(approx) != 4:
        print("No table found")
        return
    
    #sort the points
    approx = sorted(approx, key=lambda p: p[0][1]) # Sort by y-coordinate

    # Now sort left-right for the top and bottom pairs
    top_points = sorted(approx[:2], key=lambda p: p[0][0])  # Top-left, top-right
    bottom_points = sorted(approx[2:], key=lambda p: p[0][0])  # Bottom-left, bottom-right

    #want to get the ratio of distance between points to determine which way is the longer side
    top_len = np.linalg.norm(top_points[0] - top_points[1])
    side_len = np.linalg.norm(bottom_points[0] - top_points[0])
    if top_len > side_len:
        ordered_corners = np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]])
    else:
        ordered_corners = np.array([bottom_points[0], top_points[0], top_points[1], bottom_points[1]])

    #warp the image to a top down view
    table_width = int(table_length / 2) 
    dst = np.array([[0, 0], [table_length, 0], [table_length, table_width], [0, table_width]], np.float32)
    
    matrix = cv2.getPerspectiveTransform(ordered_corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, matrix, (table_length, table_width))
    
    return warped, matrix

def highlight_balls_on_table(img: np.ndarray):
    """
        Takes a preprocessed table, removes the green, converts remaining pixels to white,
        and removes small white dots below a certain area threshold
    """
    new_img = img.copy()
    hsv = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    v_channel = hsv[:, :, 2]
    mask_dark = v_channel < 17 

    # Set green areas and dark areas to black
    new_img[mask_green > 0] = [0, 0, 0]
    new_img[mask_dark] = [0, 0, 0]

    # Set other areas to white
    mask_other = (mask_green == 0) & (~mask_dark)
    new_img[mask_other] = [255, 255, 255]
    
    cv2.imshow("Mask", new_img)

    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    
    # Find contours of white regions
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Remove small white dots
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 40:
            # Fill small contours with black
            cv2.drawContours(new_img, [contour], -1, (0, 0, 0), -1)
            
    return new_img

def convolve_balls(img: np.ndarray, kernel_size: int = 5):
    """
        Convolve the image with a kernel to smooth out the balls
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    convolved_img = cv2.filter2D(img, -1, kernel)
    
    gray = cv2.cvtColor(convolved_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1600 or area < 500:
            cv2.drawContours(convolved_img, [contour], -1, (0, 0, 0), -1)
    
    return convolved_img

def convolve_balls_hough(img: np.ndarray, blur_size: int = 5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT_ALT, 
        dp=1.2,              # Resolution of accumulator
        minDist=30,        # Minimum distance between circles
        param1=40,         # Upper threshold for Canny edge detector
        param2=0.60,         # Threshold for center detection
        minRadius=5,      # Min radius to be detected
        maxRadius=100       # Max radius to be detected
    )
    

    if circles is not None:
        print(len(circles[0]), "circles found")
        output = []
        for circle in circles[0]:
            x, y, r = circle
            output.append((int(x), int(y), int(r)))
            
        return output
    
    #draw the circles on a new image
    new_img = img.copy()
    if circles is not None:
        for circle in circles[0]:
            x, y, r = circle
            cv2.circle(new_img, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.circle(new_img, (int(x), int(y)), 2, (0, 0, 255), 3) # Draw center of circle
            
    cv2.imshow("balls", new_img)
    
    return output
    