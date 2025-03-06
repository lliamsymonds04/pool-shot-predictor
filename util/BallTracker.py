import collections


class BallTracker:
    buffer_size = 3
    def __init__(self, ball_name: str):
        self.ball_name = ball_name
        
        self.buffer = collections.deque(maxlen=BallTracker.buffer_size)
        
    def update(self, x: float, y: float):
        self.buffer.append((x, y))
        
    def not_found(self):
        self.buffer.append(None)
    
    def get_position(self):
        if len(self.buffer) == 0:
            return None
        
        t = 0
        t_x = 0
        t_y = 0
        for v in self.buffer:
            if v is not None:
                t += 1
                t_x += v[0]
                t_y += v[1]
                
        if t == 0:
            return None
        else:
            return t_x / t, t_y / t