import cv2
import numpy as np

def touchup_image(img: np.ndarray):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel with the a and b channels
    limg = cv2.merge((cl, a, b))
    
    # Convert back to BGR
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # White balance - simple gray world assumption
    b, g, r = cv2.split(enhanced_img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
    
    # Find the gain of each channel
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    
    # Apply the gain
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    
    # Merge the white balanced channels
    balanced_img = cv2.merge([b, g, r])
    
    # Additional shadow removal - increase brightness in dark areas
    hsv = cv2.cvtColor(balanced_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Increase brightness in darker areas while preserving highlights
    # Simple gamma correction on V channel
    gamma = 1.2
    v_corrected = np.power(v / 255.0, 1.0 / gamma) * 255.0
    v_corrected = v_corrected.astype(np.uint8)
    
    hsv_corrected = cv2.merge([h, s, v_corrected])
    shadow_free = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)
    
    return shadow_free 