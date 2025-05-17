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
    if ball["number"] != 8 and ball["number"] != 0:
        colours[ball["name"]] = colour
    
        colour_data[ball["name"]] = {
            "hue": ball["hue"],
            "hue_range": ball["hue_range"],
            "v_range": ball["v_range"],
        } 

print(colour_data)
    
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
        
def get_colour(h: int, s: int, v: int):
    #handle black
    if s < 30 and v < 30 and (h != 0 and s != 0 and v != 0):
        return "black"
    
    #handle other colours
    for colour, values in colours.items():
        target_hue = colour_data[colour]["hue"]
        hue_range = colour_data[colour]["hue_range"]
        v_range = colour_data[colour]["v_range"]


        hue_diff = abs((int(target_hue) - int(h) + 90) % 180 - 90)
        if abs(hue_diff) > hue_range:
            continue
        
        """ 
        v_diff = abs(values[2] - int(v))
        if v_diff > 20:
            continue
         """
        if v < v_range[0] or v > v_range[1]:
            continue
              
        return colour
    #handle white
    if s < 30 and v > 200:
        return "white"
    

    return "unknown"
        
def classify_ball(x: int, y: int, r: int, hsv_image: np.ndarray, debug: bool = False):
    branches = 8
    colour_occurences = {}
    spots= 0
    for i in range(0, 360, int(360/branches)):  
        angle = np.radians(i)
        for j in range(2,r,3):
            spots +=1
            x1 = int(x + j * np.cos(angle))
            y1 = int(y + j * np.sin(angle))
            
            max_y, max_x = hsv_image.shape[:2]
            h,s,v = hsv_image[min(max(int(y1),0),max_y-1), min(max(int(x1),0),max_x-1)]
            colour = get_colour(h, s, v)
            if debug:
                print(h,s,v, colour)
            
            if colour == "unknown":
                continue
            
            colour_occurences[colour] = colour_occurences.get(colour, 0) + 1
            
    if debug:
        print(spots)
        print(colour_occurences)
     
    # Sort the colours by frequency (most frequent first)
    sorted_colours = [k for k, v in sorted(colour_occurences.items(), key=lambda x: x[1], reverse=True)]
    if "unknown" in sorted_colours:
        sorted_colours.remove("unknown")
        if len(sorted_colours) == 0:
            # no detected colours
            return "unknown"
        
    most_frequent_colour = sorted_colours[0]
    
    if most_frequent_colour == "black":
        return "black"

    if "white" in sorted_colours:
        sorted_colours.remove("white")
        if len(sorted_colours) == 0:
            return "white"
        
        most_frequent_colour = sorted_colours[0]
        if "white" != most_frequent_colour and colour_occurences["white"] > 0.1 * spots:
            return f"striped {most_frequent_colour}"
        else:
            return "white"
    
    return most_frequent_colour
        
   
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
"""                 

    colours = np.array(colours)
    mean = np.mean(colours, axis=0)
    std = np.std(colours, axis=0)
    print(std)

    threshold = 1

    distances = np.abs(colours - mean)
    mask = np.all(distances < threshold * std, axis=1)

    # Return filtered array
    filtered_colours = colours[mask]

    avg_colour = np.mean(filtered_colours, axis=0) 
    
    
    for name, data in colour_data.items():
        for colour in colours:
            hue_diff = abs((data["hue"] - colour[0] + 90) % 180 - 90)

            if hue_diff > data["hue_range"]:
                continue

        
            found_colours[name] = found_colours.get(name, 0) + 1
        
    print(found_colours)    


    return avg_colour
 """