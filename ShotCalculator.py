import cv2
import numpy as np

from MyTypes import BallClassification
from ProcessBalls import colours

def get_ghost_ball_position(ball_x: int, ball_y: int, pocket_x: int, pocket_y: int, white_ball_radius: int) -> tuple[int, int]:
    ball_to_pocket = np.array([pocket_x - ball_x, pocket_y - ball_y])
    ball_to_pocket_length = np.linalg.norm(ball_to_pocket)
    if ball_to_pocket_length == 0:
        return (ball_x, ball_y)  # Avoid division by zero

    direction = ball_to_pocket / ball_to_pocket_length
    ghost_ball_x = int(pocket_x - direction[0] * white_ball_radius)
    ghost_ball_y = int(pocket_y - direction[1] * white_ball_radius)

    return (ghost_ball_x, ghost_ball_y)
    

def does_shot_traj_collide_with_balls(ax: int, ay: int, bx: int, by: int, balls: list[tuple[int]], exclude: int) -> bool:
    #placehold
    return False

def calculate_best_shot(table: np.ndarray, balls: list[tuple[int]], ball_classifications: list[BallClassification], stripped: bool = False):
    print("Calculating best shot...")

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
    for i, classification in enumerate(ball_classifications):
        if classification.colour != "white" and (not stripped or classification.stripped):
            possible_balls.append(i)

    if len(possible_balls) == 0:
        print("No possible balls to hit!")
        return

    #calculate the best shot
    white_ball_x, white_ball_y, white_ball_r = balls[white_ball_index]
    best_shot_index = -1
    best_pocket_index = -1
    best_shot_angle = float('inf')  # Start with a very large angle
    for pocket in pocket_positions:
        pocket_x, pocket_y = pocket

        for ball_index in possible_balls:
            ball_x, ball_y, _ = balls[ball_index]
            ghost_ball_x, ghost_ball_y = get_ghost_ball_position(ball_x, ball_y, pocket_x, pocket_y, white_ball_r)

            #determine if the white ball can hit the ghost ball
            
            #determine if the shot trajectory collides with any other balls

            #determine shot angle
            
            #if shot angle is less than curreent best angle, update best shot
            
    #draw the best shot
    



        
    
    
    cv2.imshow("Best Shot", table)
    