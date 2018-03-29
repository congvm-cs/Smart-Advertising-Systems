import imutils
import dlib
import numpy as np 
import argparse
from imutils import face_utils
# from keras.models import load_model
import cv2

# import skvideo.io
cap = cv2.VideoCapture(0)

# camera = cv2.VideoCapture(0)

if cap.isOpened() is False:
    print("Cannot open camera")
    exit()

# model_path = '/mnt/e/Smart-Advertising-Systems-master/Models/weights-improvement-23-0.23-0.92.hdf5'
# model_path = 'E:/Smart-Advertising-Systems-master/Models/weights-improvement-23-0.23-0.92.hdf5'

detector = dlib.get_frontal_face_detector()

# model = load_model(model_path)

while True:
    # (rval, frame) = camera.read()
    ret, frame = cap.read()

    if ret is True:
        image = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            (x, y, h, w) = face_utils.rect_to_bb(rect)
            
            face_rect = gray[y:y+w, x:x+h]
            cv2.imshow("Face", face_rect)

            cv2.putText(image, "Face #{}".format(i+1), (x-10, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)
            
            cv2.circle(image, (x, y), 5, (255, 0, 255))
            cv2.circle(image, (x + w, y + h), 5, (255, 255, 255), -1)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # cv2.imwrite()
        cv2.imshow('detect', image)
        key = cv2.waitKey(1)
        if key == 27:
            break    
    else:
        print("Cannot open camera")
    
    
