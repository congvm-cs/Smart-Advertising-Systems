import cv2
import requests

information = {}
def check_connect():
    connect_check = requests.get('http://127.0.0.1:5000/connected')
    if connect_check.text == 'Connected':
        return True
    else:
        return False
def get_video_index():
    
    return video_index

def get_frame():
    # cap.set(CV_CAP_PROP_FPS, 20)
    
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (128,128))
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret:
        print("Error: failed to capture image")
        break
    # image_arr.append(frame)
    #cv2.imshow('gray', gray)
    # byte_img = image.tobytes()
    cv2.waitKey(50)
    return frame
    
def __main__():
    cap = cv2.VideoCapture(0)

    while True:
        
    
        information['index'] = get_video_index()
        information['image'] = 


        
