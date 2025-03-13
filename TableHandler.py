import cv2
from util.TouchupImage import touchup_image 


#load an image
image_name = "broken_1"
image_path = f"images/table/{image_name}.jpg"
image = cv2.imread(image_path)

if image is not None:
    touchedup_image = touchup_image(image)

    #find contours of table


    #warp the image to a top down view
    
    #find the balls
    
    #scale the image
    w = 800
    old_w = touchedup_image.shape[1]
    scale = w / old_w
    h = int(touchedup_image.shape[0] * scale)
    final_image = cv2.resize(touchedup_image, (w, h))

    #display the image
    cv2.imshow("Table", final_image)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
else:
    print("Image not found")