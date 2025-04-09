import numpy as np
import cv2

# lower_green = np.array([35, 40, 40])
# upper_green = np.array([90, 255, 255])
lower_green = np.array([30, 20, 40])
upper_green = np.array([100, 255, 255])


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
    mask = cv2.inRange(hsv, lower_green, upper_green)
    new_img[mask > 0] = [0, 0, 0]  # Set green areas to black
    new_img[mask == 0] = [255, 255, 255]  # Set non-green areas to white
    
    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    
    # Find contours of white regions
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Remove small white dots
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10:
            # Fill small contours with black
            cv2.drawContours(new_img, [contour], -1, (0, 0, 0), -1)
            
    return new_img