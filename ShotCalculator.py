import cv2
import numpy as np

from MyTypes import BallClassification
from ProcessBalls import colours

cue_colour = (255, 235, 126)
ball_traj_colour = (255,255,255)
ghost_ball_colour = (235, 236,240)

cue_thickness = 5
ball_traj_thickness = 2

#convert to BGR
cue_colour = cue_colour[::-1]  # Reverse the tuple to convert to BGR
ball_traj_colour = ball_traj_colour[::-1]  # Reverse the tuple to convert to BGR
ghost_ball_colour = ghost_ball_colour[::-1]  # Reverse the tuple to convert to BGR

def get_traj(ax: int, ay: int, bx: int, by: int):
    v = np.array([bx - ax, by - ay])
    length = np.linalg.norm(v)
    if length == 0:
        return np.array([0, 0])
    v /= length  # Normalize the vector
    
    return v

def get_ghost_ball_position(ball_x: int, ball_y: int, pocket_x: int, pocket_y: int, ball_radius: int, white_ball_radius: int) -> tuple[int, int]:
    ball_to_pocket = np.array([pocket_x - ball_x, pocket_y - ball_y])
    ball_to_pocket_length = np.linalg.norm(ball_to_pocket)
    if ball_to_pocket_length == 0:
        return (ball_x, ball_y)  # Avoid division by zero

    direction = ball_to_pocket / ball_to_pocket_length
    r = white_ball_radius + ball_radius
    ghost_ball_x = int(ball_x - direction[0] * r)
    ghost_ball_y = int(ball_y - direction[1] * r)

    return (ghost_ball_x, ghost_ball_y)
    
def line_intersects_circle(x1, y1, x2, y2, cx, cy, r):
    """Check if line segment from (x1,y1) to (x2,y2) intersects circle at (cx,cy) with radius r"""
    # Vector from start to end of line
    dx = x2 - x1
    dy = y2 - y1
    
    # Vector from start of line to circle center
    fx = x1 - cx
    fy = y1 - cy
    
    # Quadratic equation coefficients
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - r * r
    
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return False  # No intersection
    
    # Check if intersection points are within the line segment
    discriminant = discriminant ** 0.5
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)
    
    # If either intersection point is within [0,1], there's a collision
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

def does_shot_collide_with_balls(ball_index, x: int, y: int, balls: list[tuple[int]]) -> bool:
    #placehold
    ball_x, ball_y, ball_r = balls[ball_index]
    #shot_traj = get_traj(x, y, ball_x, ball_y)
    v = np.array([ball_x - x, ball_y - y], dtype=float)
    shot_length = np.linalg.norm(v)
    
    #normalize the vector
    if shot_length == 0:
        return False
    v /= shot_length  # Normalize the vector
    
    #get the perpendicular vector
    perp_v = np.array([-v[1], v[0]])  # Rotate 90 degrees to get the perpendicular vector
    
    pos_a = np.array([ball_x + ball_r * perp_v[0], ball_y + ball_r * perp_v[1]], dtype=float)
    pos_b = np.array([ball_x - ball_r * perp_v[0], ball_y - ball_r * perp_v[1]], dtype=float)
    
    
    for i, (other_x, other_y, other_r) in enumerate(balls):
        if i == ball_index:
            continue
            
        # check if the ball intersects with the shot trajectory
        if line_intersects_circle(pos_a[0], pos_a[1], x, y, other_x, other_y, other_r):
            return True
        if line_intersects_circle(pos_b[0], pos_b[1], x, y, other_x, other_y, other_r):
            return True

        
    return False

def calculate_best_shot(table: np.ndarray, balls: list[tuple[int]], ball_classifications: list[BallClassification], stripped: bool = False):
    if stripped:
        print("Calculating best shot for stripes...")
    else:
        print("Calculating best shot for solids...")

    #draw the balls
    for ball, classification in zip(balls, ball_classifications):
        x, y, radius = ball
        hsv = colours[classification.colour]
        b, g, r = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        #convert to int
        b, g, r = int(b), int(g), int(r)
        
        cv2.circle(table, (x, y), radius, (b,g,r), 2)

    #find the white ball
    white_ball_index = -1
    for ball, classification in zip(balls, ball_classifications):
        if classification.colour == "white":
            white_ball_index = balls.index(ball)
            break

    if white_ball_index == -1:
        print("No white ball found!")
        return

    #create the pocket positions
    table_width, table_length = table.shape[:2]
    pocket_positions = [
        (0,0), #top left
        (int(table_length / 2), 0), #top middle
        (table_length, 0), #top right
        (0, table_width), #bottom left
        (int(table_length / 2), table_width), #bottom middle
        (table_length, table_width) #bottom right
    ]

    #get all the possible balls
    possible_balls = []
    found_black = -1
    for i, classification in enumerate(ball_classifications):
        if i == "unknown":
            continue
        elif classification.colour == "black":
            found_black = True
        elif classification.colour != "white" and (classification.stripped == stripped):
            possible_balls.append(i)

    if len(possible_balls) == 0:
        #only the black ball is left can it be sunk
        if found_black >= 0:
            possible_balls.append(found_black)
        else:
            print("No possible balls to hit!")
            return

    #calculate the best shot
    white_ball_x, white_ball_y, white_ball_r = balls[white_ball_index]
    best_shot_index = -1
    best_pocket_index = -1
    best_shot_angle = float('inf')  # Start with a very large angle
    # best_shot_angle = 0
    for pocket in pocket_positions:
        pocket_x, pocket_y = pocket

        for ball_index in possible_balls:
            ball_x, ball_y, ball_r = balls[ball_index]
            ghost_ball_x, ghost_ball_y = get_ghost_ball_position(ball_x, ball_y, pocket_x, pocket_y, ball_r, white_ball_r)

            #determine if the white ball can hit the ghost ball
            if does_shot_collide_with_balls(white_ball_index, ghost_ball_x, ghost_ball_y, balls):
                continue

            #determine if the shot trajectory collides with any other balls
            if does_shot_collide_with_balls(ball_index, pocket_x, pocket_y, balls):
                continue
            
            #determine shot angle
            traj_x = ghost_ball_x - white_ball_x
            traj_y = ghost_ball_y - white_ball_y
            white_ball_traj = np.array([traj_x, traj_y], dtype=float)
            traj_length = np.linalg.norm(white_ball_traj)
            if traj_length == 0:
                continue
            white_ball_traj /= traj_length
            pocket_traj = np.array([pocket_x - ghost_ball_x, pocket_y - ghost_ball_y], dtype=float)
            pocket_traj_length = np.linalg.norm(pocket_traj)
            if pocket_traj_length == 0:
                continue
            pocket_traj /= pocket_traj_length

            shot_angle = abs(np.arccos(np.clip(np.dot(white_ball_traj, pocket_traj), -1.0, 1.0)) * (180 / np.pi))
            
            
            #if shot angle is less than curreent best angle, update best shot
            if shot_angle < best_shot_angle:
                best_shot_angle = shot_angle
                best_shot_index = ball_index
                best_pocket_index = pocket_positions.index(pocket)
            
    #draw the best shot
    if best_shot_index == -1:
        print("No best shot found!")
        return
    
    best_ball = balls[best_shot_index]
    best_ball_x, best_ball_y, best_ball_r = best_ball
    best_pocket = pocket_positions[best_pocket_index]
    best_pocket_x, best_pocket_y = best_pocket
    ghost_ball_x, ghost_ball_y = get_ghost_ball_position(best_ball_x, best_ball_y, best_pocket_x, best_pocket_y, best_ball_r, white_ball_r)

    #draw the ghost ball
    cv2.circle(table, (ghost_ball_x, ghost_ball_y), white_ball_r, ghost_ball_colour, -1)
    
    #draw the strike line
    cv2.line(table, (white_ball_x, white_ball_y), (ghost_ball_x, ghost_ball_y), cue_colour, cue_thickness)

    #draw the ball trajectory
    cv2.line(table, (best_ball_x, best_ball_y), (best_pocket_x, best_pocket_y), ball_traj_colour, ball_traj_thickness)
    
    #output the image
    cv2.imshow("Best Shot", table)

    
    #output the best shot
    
    #determine pocket position
    pocket_position = "top left" if best_pocket_index == 0 else \
                     "top middle" if best_pocket_index == 1 else \
                     "top right" if best_pocket_index == 2 else \
                     "bottom left" if best_pocket_index == 3 else \
                     "bottom middle" if best_pocket_index == 4 else \
                     "bottom right"

    print(f"Hit the {ball_classifications[best_shot_index].colour} ball into the {pocket_position} pocket.")