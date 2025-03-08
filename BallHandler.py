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
          
        for ball in ball_data["balls"]:
            colour: list[int] = ball["colour"]
            hsv_colour = np.array([[colour]], dtype=np.uint8)  
            bgr = cv2.cvtColor(hsv_colour, cv2.COLOR_HSV2BGR)[0][0].tolist()

            #check if ball can be striped
            if ball["number"] != 8 and ball["number"] != 0:
                self.balls["half_" + ball["name"]] = BallTracker(ball_name=ball["name"], ball_number=ball["number"] + 8, colour=bgr)
                self.colours[ball["name"]] = colour
            

            self.balls[ball["name"]] = BallTracker(ball_name=ball["name"], ball_number=ball["number"], colour=bgr)
            
        
    def eval_circle(self, x: float, y: float, r: float, hsv_frame: np.ndarray):
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
    
    def update_ball(self, x: float, y: float, r: float, ball_name: str):
        self.balls[ball_name].update(x, y, r)
    
    def not_found(self, ball_name: str):
        self.balls[ball_name].not_found()     
    
    def draw_balls(self, frame: np.ndarray):
       for ball in self.balls.values():
            ball.draw(frame)