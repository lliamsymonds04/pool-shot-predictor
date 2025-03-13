import cv2
import numpy as np
from util.TouchupImage import touchup_image 


#green color range
lower_green = np.array([35, 40, 40])
upper_green = np.array([90, 255, 255])


def detect_pool_table(image_path: str, debug: bool = False):
    image = cv2.imread(image_path)

    if image is not None:
        touchedup_image = touchup_image(image)
        hsv = cv2.cvtColor(touchedup_image, cv2.COLOR_BGR2HSV)

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
        
        if debug:
            cv2.drawContours(touchedup_image, [largest_contour], -1, (255, 0, 0), 10)
            cv2.drawContours(touchedup_image, [hull], -1, (0, 255, 0), 10)
        
        perimeter = cv2.arcLength(hull, True)
        epsilon_factor = 0.02
        #adjust the epsilon factor to get a 4 sided polygon
        for i in range(10):
            approx = cv2.approxPolyDP(hull, epsilon_factor * perimeter, True)
            if len(approx) == 4:
                break     
            
            epsilon_factor += 0.002

        if len(approx) != 4:
            print("No table found")
            return
        
        # cv2.drawContours(touchedup_image, [approx], -1, (0, 0, 255), 10)
        
        #sort the points
        # Sort by Y first (top 2 points, bottom 2 points)
        approx = sorted(approx, key=lambda p: p[0][1])

        # Now sort left-right for the top and bottom pairs
        top_points = sorted(approx[:2], key=lambda p: p[0][0])  # Top-left, top-right
        bottom_points = sorted(approx[2:], key=lambda p: p[0][0])  # Bottom-left, bottom-right

        ordered_corners = np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]])

        cv2.drawContours(touchedup_image, [ordered_corners.astype(int)], -1, (255, 0, 255), 10)
        cv2.circle(touchedup_image, (ordered_corners[0][0]).astype(int), 40, (0, 0, 255), -1)
        
        #warp the image to a top down view
        
        #find the balls
        
        #scale the final image
        w = 800
        old_w = touchedup_image.shape[1]
        scale = w / old_w
        h = int(touchedup_image.shape[0] * scale)
        final_image = cv2.resize(touchedup_image, (w, h))
        # final_image = cv2.resize(mask, (w, h))

        #display the image
        cv2.imshow("Table", final_image)
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()

    else:
        print("Image not found")


if __name__ == "__main__":
    image_name = "broken_1"
    image_path = f"images/table/{image_name}.jpg"

    detect_pool_table(image_path)
    
