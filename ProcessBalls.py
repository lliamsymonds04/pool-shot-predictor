import cv2
import json
import numpy as np
from dataclasses import dataclass

with open('data/PoolBalls.json', 'r') as f:
    ball_data = json.load(f)
    
colours = {}
colour_data = {}

BRIGHT_V = 200
DARK_V = 30
COLOUR_SAT_THREASHOLD = 30
HIGHER_SAT_THREASHOLD = 70

@dataclass
class BallClassification:
    colour: str
    striped: bool
          
for ball in ball_data["balls"]:
    colour: list[int] = ball["colour"]

    #check if ball can be striped
    colours[ball["name"]] = colour
    if ball["number"] != 8 and ball["number"] != 0:
        colour_data[ball["name"]] = {
            "hue": ball["hue"],
            "hue_range": ball["hue_range"],
            "v_range": ball["v_range"],
            "sat_range": ball["sat_range"]
        } 
    
def draw_balls_debug(img: np.ndarray, balls: list[tuple[int]]):
    """
    draws the balls on the image
    """
    new_img = img.copy()
    for ball in balls:
        x, y, r = ball
        cv2.circle(new_img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(new_img, (x, y), 2, (0, 0, 255), 3) # Draw center of circle
    return new_img

def classify_balls(balls: list[tuple[int]], table_img: np.ndarray):
    
    classifications = []
    
    hsv_image = cv2.cvtColor(table_img, cv2.COLOR_BGR2HSV)
    for ball in balls:
        x,y,r = ball
        
        classifications.append(classify_ball(x, y, r, hsv_image))

    return classifications
        
def get_colour(h: int, s: int, v: int):
    #handle black
    if s < COLOUR_SAT_THREASHOLD and v < DARK_V and (h != 0 and s != 0 and v != 0):
        return "black"

    #handle white
    if s < COLOUR_SAT_THREASHOLD and v > BRIGHT_V:
        return "white"

    #handle other colours
    for colour in colour_data.keys():
        target_hue = colour_data[colour]["hue"]
        hue_range = colour_data[colour]["hue_range"]
        v_range = colour_data[colour]["v_range"]
        sat_range = colour_data[colour]["sat_range"]

        hue_diff = abs((int(target_hue) - int(h) + 90) % 180 - 90)
        if abs(hue_diff) > hue_range:
            continue
        
        if v < v_range[0] or v > v_range[1]:
            continue

        if s < sat_range[0] or s > sat_range[1]:
            continue
              
        return colour

    #classify white again but with higher s
    if s < HIGHER_SAT_THREASHOLD and v > BRIGHT_V:
        return "white"
   
    return "unknown"
        
def classify_ball(x: int, y: int, r: int, hsv_image: np.ndarray, debug: bool = False) -> BallClassification:
    branches = 8
    colour_occurrences = {}
    spots= 0
    max_y, max_x = hsv_image.shape[:2]
    for i in range(0, 360, int(360/branches)):  
        angle = np.radians(i)
        for j in range(4,r,2):
            spots +=1
            x1 = int(x + j * np.cos(angle))
            y1 = int(y + j * np.sin(angle))
            
            h,s,v = hsv_image[min(max(int(y1),0),max_y-1), min(max(int(x1),0),max_x-1)]
            colour = get_colour(h, s, v)
            if debug:
                print(h,s,v, colour)
            
            if colour == "unknown":
                continue
            
            colour_occurrences[colour] = colour_occurrences.get(colour, 0) + 1

    # Sort the colours by frequency (most frequent first)
    sorted_colours = [k for k, v in sorted(colour_occurrences.items(), key=lambda x: x[1], reverse=True)]
    if "unknown" in sorted_colours:
        sorted_colours.remove("unknown")
        
    if len(sorted_colours) == 0:
        # no detected colours
        return BallClassification(colour="unknown", striped=False)

    if debug:
        print(spots)
        print(colour_occurrences)   

    most_frequent_colour = sorted_colours[0]
    
    if most_frequent_colour == "black":
        return BallClassification(colour="black", striped=False)

    if "white" in sorted_colours:
        sorted_colours.remove("white")
        if len(sorted_colours) == 0:
            return BallClassification(colour="white", striped=False)
        
        most_frequent_colour = sorted_colours[0]
        if colour_occurrences["white"] > 2:
            return BallClassification(colour=most_frequent_colour, striped=True)
    
    return BallClassification(colour=most_frequent_colour, striped=False)