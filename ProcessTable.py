import numpy as np
import cv2

# lower_green = np.array([50, 40, 20])
lower_green = np.array([50, 10, 30])
upper_green = np.array([85, 200, 255])

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
    
    return matrix
           
def warp_table(img: np.ndarray, warp_matrix: np.ndarray, table_length: int = 800):
    table_width = int(table_length / 2)
    
    return cv2.warpPerspective(img, warp_matrix, (table_length, table_width))

def remove_table_green(img: np.ndarray):
    """
    removes the green from the table
    """
    new_img = img.copy()
    hsv = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
 
    v_channel = hsv[:, :, 2]
    mask_dark = v_channel < 17

    combined_mask = cv2.bitwise_or(mask_green, mask_dark.astype(np.uint8) * 255)  
    # Set green areas to black
    new_img[combined_mask > 0] = [0, 0, 0]
    

    #use contouring to remove small white dots
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 40:
            # Fill small contours with black
            cv2.drawContours(new_img, [contour], -1, (0, 0, 0), -1)
    
    return new_img


def make_balls_white(img: np.ndarray):
    """
    converts all non-black pixels to white
    """
    new_img = img.copy()
    hsv = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
    # Set all non-black pixels to white
    mask_non_black = np.any(hsv != [0, 0, 0], axis=2)
    new_img[mask_non_black] = [255, 255, 255]
    return new_img


def find_balls(img: np.ndarray, blur_size: int = 5):
    """
    takes the pre-highlighted balls and returns a tupe of (x,y,r) for each ball
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT_ALT, 
        dp=1.2,              # Resolution of accumulator
        minDist=30,        # Minimum distance between circles
        param1=50,         # Upper threshold for Canny edge detector
        param2=0.60,         # Threshold for center detection
        minRadius=5,      # Min radius to be detected
        maxRadius=100       # Max radius to be detected
    )

    new_img = img.copy()
    if circles is not None:
        output = []
        for circle in circles[0]:
            x, y, r = circle
            output.append((int(x), int(y), int(r)))
            cv2.circle(new_img, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.circle(new_img, (int(x), int(y)), 2, (0, 0, 255), 3) # Draw center of circle

    return output, new_img
    

def merge_balls(balls1: list[tuple[int]], balls2: list[tuple[int]]):
    """
    merges the two lists of balls
    """
    merged_balls = []
    ball1_list = [False] * len(balls1)
    ball2_list = [False] * len(balls2)
       
    i,j = 0, 0
    while i < len(balls1) and j < len(balls2):
        b1 = balls1[i]
        b2 = balls2[j]
        x1, y1, r1 = b1
        x2, y2, r2 = b2
        dist = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
        if dist < (r1 + r2) / 2:
            #the two balls are close enough to be merged
            merged_balls.append((int((x1 + x2) / 2), int((y1 + y2) / 2), max(r1, r2)))
            ball1_list[i] = True
            ball2_list[j] = True

            i += 1
            j = 0
        elif j < len(balls2) - 1:
            j += 1
        else:
            i += 1
            j = 0
            
    #add the remaining balls
    for i in range(len(balls1)):
        if not ball1_list[i]:
            merged_balls.append(balls1[i])
            
    for j in range(len(balls2)):
        if not ball2_list[j]:
            merged_balls.append(balls2[j])
            
    return merged_balls