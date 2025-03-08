import collections
from typing import List
import cv2


class BallTracker:
    position_buffer_size = 2
    radius_buffer_size = 2
    def __init__(self, ball_name: str, ball_number: int, colour: List[int]):
        self.ball_number = ball_number
        self.ball_name = ball_name
        self.colour = colour
        self.is_stripped = False        
        
        if self.ball_name == "black":
            self.ball_text = "8-ball"
        elif self.ball_name == "white":
            self.ball_text = "cue"
        elif self.ball_number <= 7:
            self.ball_text = str(self.ball_number)
        else:
            self.ball_text = str(self.ball_number)
            self.is_stripped = True
        
        self.position_buffer = collections.deque(maxlen=BallTracker.position_buffer_size)
        self.radius_buffer = collections.deque(maxlen=BallTracker.radius_buffer_size)
        
    def update(self, x: float, y: float, radius: float):
        self.position_buffer.append((x, y))
        self.radius_buffer.append(radius)
        
    def not_found(self):
        if len(self.position_buffer) > 0:
            self.position_buffer.popleft()
        if len(self.radius_buffer) > 0:
            self.radius_buffer.popleft()

    
    def get_position(self):
        if len(self.position_buffer) == 0:
            return None
        
        t = 0
        t_x = 0
        t_y = 0
        for v in self.position_buffer:
            if v is not None:
                t += 1
                t_x += v[0]
                t_y += v[1]
                
        if t == 0:
            return None
        else:
            return int(t_x / t), int(t_y / t)
        
    def get_radius(self):
        if len(self.radius_buffer) == 0:
            return None
        
        t = 0
        t_r = 0
        for v in self.radius_buffer:
            if v is not None:
                t += 1
                t_r += v
                
        if t == 0:
            return None
        else:
            return int(t_r / t)
        
    def draw(self, frame):
        pos = self.get_position()
        r = self.get_radius()

        if pos is not None and r is not None:
            cv2.circle(frame, pos, r, self.colour, 4)
            if self.is_stripped:
                cv2.circle(frame, pos, r + 4, [255,255,255],2)
                
            #calc text pos
            r_mult = 1.5
            text_pos = (int(pos[0] + r * r_mult), int(pos[1] + r * r_mult))
            cv2.putText(frame, self.ball_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)