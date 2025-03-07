import collections


class BallTracker:
    position_buffer_size = 5
    radius_buffer_size = 2
    def __init__(self, ball_name: str, ball_number: int):
        self.ball_number = ball_number
        self.ball_name = ball_name
        
        self.position_buffer = collections.deque(maxlen=BallTracker.position_buffer_size)
        self.radius_buffer = collections.deque(maxlen=BallTracker.radius_buffer_size)
        
    def update(self, x: float, y: float, radius: float):
        self.position_buffer.append((x, y))
        self.radius_buffer.append(radius)
        
    def not_found(self):
        self.position_buffer.append(None)
        self.radius_buffer.append(None)
    
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