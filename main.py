import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
    
    circles = cv2.HoughCircles(blurred_frame, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=40, param2=30, minRadius=10, maxRadius=40)
    
    if circles is not None:
        for circle in circles[0, :]:
            x, y, r = circle
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 4) 
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()