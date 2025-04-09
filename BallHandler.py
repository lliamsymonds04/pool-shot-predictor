import json
import cv2
import numpy as np
from util.BallTracker import BallTracker


with open('data/PoolBalls.json', 'r') as f:
    ball_data = json.load(f)
    

class BallHandler:
    def __init__(self):
        self.balls = {}
        self.colours = {}
        self.colour_data = {}
          
        for ball in ball_data["balls"]:
            colour: list[int] = ball["colour"]
            hsv_colour = np.array([[colour]], dtype=np.uint8)  
            bgr = cv2.cvtColor(hsv_colour, cv2.COLOR_HSV2BGR)[0][0].tolist()

            #check if ball can be striped
            if ball["number"] != 8 and ball["number"] != 0:
                self.balls["half_" + ball["name"]] = BallTracker(ball_name=ball["name"], ball_number=ball["number"] + 8, colour=bgr)
                self.colours[ball["name"]] = colour
            
                self.colour_data[ball["name"]] = {
                    "hue": ball["hue"],
                    "hue_range": ball["hue_range"]
                } 

            self.balls[ball["name"]] = BallTracker(ball_name=ball["name"], ball_number=ball["number"], colour=bgr)
            
        
    def classify_ball(self, x: float, y: float, r: float, hsv_frame: np.ndarray):
        #find the correspoding colour 
        found_colours = []
        points = [
            (x, y),
            (x + r, y),
            (x, y + r),
            (x - r, y),
            (x, y + r)
        ]
        max_y, max_x = hsv_frame.shape[:2]
        hsv_values = [hsv_frame[min(max(int(y),0),max_y-1), min(max(int(x),0),max_x-1)] for x, y in points]
        
        arr = []
        for v in hsv_values:
           arr.append([v[0].astype(int), v[1], v[2]])
       
        found_black = False
        found_white = False 
        found_colour = ""
        
        found_colours = {}
        for point in hsv_values:
            #check for black
            if int(point[2]) < 30: 
                found_black = True
                continue
            
            found_colour = False
            for colour, values in self.colours.items():
                if int(values[1]) < 50: #if the saturation is low, the colour is not valid
                    continue
               
                hue_diff = abs((values[0] - point[0].astype(int) + 90) % 180 - 90)
                if abs(hue_diff) > 6:
                    continue
                
                v = point[2].astype(int)
                if v == 255:
                    if values[2] < 170:
                        continue
                else:
                    v_diff = abs(values[2] - point[2].astype(int))
                    if v_diff > 20:
                        continue
                
                found_colours[colour] = found_colours.get(colour, 0) + 1
                found_colour = True
                
            #check for white
            if not found_colour:
                if int(point[1]) < 80 and int(point[2]) > 100:
                    found_white = True
        
        
        if found_colours:
            found_colour = max(found_colours, key=found_colours.get)
        else:
            found_colour = ""
            
        if found_colour != "":
            if found_white:
                return "half_" + found_colour
            else:
                return found_colour
        elif found_white:
            return "white"
        elif found_black:
            return "black"
        else:
            return None
    
    def classify_ball_2(self, x: float, y: float, r: float, hsv_image: np.ndarray):
        # Create a mask for the circle
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x,y), r, 255, -1)
        
        # Apply the mask to the HSV image
        masked_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
        
        # Create a saturation mask for pixels above the minimum saturation
        colour_sat_mask = cv2.inRange(masked_image, np.array([0, 140, 50]), np.array([179, 255, 255]))
        # low_sat_mask = cv2.inRange(masked_image, np.array([0, 0, 0]), np.array([179, 30, 255]))
        # Combine both masks
        colour_final_mask = cv2.bitwise_and(mask, colour_sat_mask)
        # low_final_mask = cv2.bitwise_and(mask, low_sat_mask)

        
        # Get all non-zero pixels
        non_zero_pixels = hsv_image[colour_final_mask > 0]
        
        # If no pixels meet the criteria, return None
        if len(non_zero_pixels) == 0:
            return None
        
        # Calculate the average color
        avg_color = np.mean(non_zero_pixels, axis=0)
        
        return tuple(avg_color.astype(int))
    
    def classify_ball_3(self, x: float, y: float, r: float, hsv_image: np.ndarray):
        branches = 8
        found_colours = {} 
        colours = []
        for i in range(0, 360, int(360/branches)):  
            angle = np.radians(i)
            for j in range(2,r,3):
                x1 = int(x + j * np.cos(angle))
                y1 = int(y + j * np.sin(angle))
                
                max_y, max_x = hsv_image.shape[:2]
                h,s,v = hsv_image[min(max(int(y1),0),max_y-1), min(max(int(x1),0),max_x-1)]
                
                if s < 30:
                    continue
                
                # if v < 40:
                    # continue
                
                colours.append((h,s,v))

                """
                for colour, values in self.colours.items():
                    hue_diff = abs((values[0] - h + 90) % 180 - 90)
                    if abs(hue_diff) > 6:
                        continue
                    
                    v_diff = abs(values[2] - v)
                    if v_diff > 20:
                        continue
                    
                    # return colour
                    found_colours[colour] = found_colours.get(colour, 0) + 1
                """    
                    
        colours = np.array(colours)
        mean = np.mean(colours, axis=0)
        std = np.std(colours, axis=0)

        threshold = 1

        distances = np.abs(colours - mean)
        mask = np.all(distances < threshold * std, axis=1)
    
        # Return filtered array
        filtered_colours = colours[mask]
    
        avg_colour = np.mean(filtered_colours, axis=0) 
        
        
        for name, data in self.colour_data.items():
            for colour in colours:
                hue_diff = abs((data["hue"] - colour[0] + 90) % 180 - 90)

                if hue_diff > data["hue_range"]:
                    continue

            
                found_colours[name] = found_colours.get(name, 0) + 1
            
        print(avg_colour)
        print(found_colours)    


        return avg_colour

    
    def update_ball(self, x: float, y: float, r: float, ball_name: str):
        self.balls[ball_name].update(x, y, r)
    
    def not_found(self, ball_name: str):
        self.balls[ball_name].not_found()     
    
    def draw_balls(self, frame: np.ndarray):
       for ball in self.balls.values():
            ball.draw(frame)