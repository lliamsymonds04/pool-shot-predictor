import cv2
import numpy as np

def plot_hsv(frame: np.ndarray, x: float, y: float, y_offset: int = 0):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    point = hsv[int(y), int(x)]
    h, s, v = point
    
    
    cv2.putText(frame, f"H: {h}, S: {s}, V: {v}", (int(x), int(y) + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)