import cv2
import requests

image_arr = []

cap = cv2.VideoCapture(0)
# cap.set(CV_CAP_PROP_FPS, 20)
while True: 
    ret, frame = cap.read()
    frame = cv2.resize(frame, (128,128))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret:
        print("Error: failed to capture image")
        break
    # image_arr.append(frame)
    cv2.imshow('gray', gray)
    cv2.waitKey(50)
cap.release()
cv2.destroyAllWindows()

    # np.save('image_mat', image_arr)
    # upload = requests.post()


        
