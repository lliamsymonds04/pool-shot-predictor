import cv2
import json
import numpy as np

with open('data/PoolBalls.json', 'r') as f:
    ball_data = json.load(f)
    
colours = {}
colour_data = {}
          
for ball in ball_data["balls"]:
    colour: list[int] = ball["colour"]

    #check if ball can be striped
    colours[ball["name"]] = colour
    if ball["number"] != 8 and ball["number"] != 0:
    
        colour_data[ball["name"]] = {
            "hue": ball["hue"],
            "hue_range": ball["hue_range"],
            "v_range": ball["v_range"],
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
    if s < 30 and v < 30 and (h != 0 and s != 0 and v != 0):
        return "black"

    #handle white
    if s < 30 and v > 200:
        return "white"

    #handle other colours
    for colour in colour_data.keys():
        target_hue = colour_data[colour]["hue"]
        hue_range = colour_data[colour]["hue_range"]
        v_range = colour_data[colour]["v_range"]

        hue_diff = abs((int(target_hue) - int(h) + 90) % 180 - 90)
        if abs(hue_diff) > hue_range:
            continue
        
        if v < v_range[0] or v > v_range[1]:
            continue
              
        return colour

    #classify white again but with higher s
    if s < 50 and v > 200:
        return "white"
   
    return "unknown"
        
def classify_ball(x: int, y: int, r: int, hsv_image: np.ndarray, debug: bool = False) -> tuple[str, bool]:
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
            return ("unknown", False)

    if debug:
        print(spots)
        print(colour_occurrences)   

    most_frequent_colour = sorted_colours[0]
    
    if most_frequent_colour == "black":
        return ("black", False)

    if "white" in sorted_colours:
        sorted_colours.remove("white")
        if len(sorted_colours) == 0:
            return ("white", False)
        
        most_frequent_colour = sorted_colours[0]
        if "white" != most_frequent_colour and colour_occurrences["white"] >= 2:
            return (most_frequent_colour, True)
        else:
            return ("white", False)
    
    return (most_frequent_colour, False)
        
   
def debug_classify_ball(balls: list[tuple[int]], table_img: np.ndarray, index: int):
    ball = balls[index]
    x,y,r = ball
    hsv_image = cv2.cvtColor(table_img, cv2.COLOR_BGR2HSV)
    result = classify_ball(x, y, r, hsv_image, True)

    
    debug_img = table_img.copy()
    #draw the ball
    cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
    cv2.circle(debug_img, (x, y), 2, (0, 0, 255), 3) # Draw center of circle
    
    cv2.imshow("Debug", debug_img)
    return result

RING_THICKNESS = 3
WHITE_RING_THICKNESS = 2
RING_OFFSET = 2
def draw_ball(x: int, y: int, radius: int, colour: str, stripped: bool, number: int, img: np.ndarray):
    """
    Draws the ball on the image
    """
    #get the hsv colour of the ball
    hsv = colours[colour]
    #convert to bgr
    b, g, r = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0]
    
    # Convert numpy.uint8 to Python int for OpenCV
    ring_colour = (int(b), int(g), int(r))
    #draw the coloured circle
    radius += RING_OFFSET
    cv2.circle(img, (x, y), radius, ring_colour, RING_THICKNESS)
    cv2.putText(img, str(number), (x + int(radius * 1.5), int(y - radius * 1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if stripped:
        cv2.circle(img, (x, y), radius + RING_THICKNESS, (255, 255, 255), WHITE_RING_THICKNESS)
        
    return img

def draw_balls_classificiation(table_img: np.ndarray, balls: list[tuple[int]], classifications: list[tuple[str, bool]]):
    """
    Draws the balls on the image
    """
    new_img = table_img.copy()
    for i, ball in enumerate(balls):
        x, y, r = ball
        colour, stripped = classifications[i]
        new_img = draw_ball(x, y, r, colour, stripped, i, new_img)
        
    return new_img