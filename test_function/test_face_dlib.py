import imutils
import dlib
import numpy as np 
from imutils.face_utils import rect_to_bb, FaceAligner
# from keras.models import load_model
import cv2

# cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

file_path = '/media/vmc/12D37C49724FE954/Face_Data/CAS-PEAL-R1/FRONTAL/Accessory/FO_000270_IEU+00_PM+00_EN_A1_D0_T0_BB_M0_R1_S0.tif'
# while True:
# ret, frame = cap.read()
frame = cv2.imread(file_path)
frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

for rect in rects:
    (x, y, w, h) = rect_to_bb(rect)             # Positions of rectangle contains face

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    offset = int(0.2*x) 
    x = x - offset
    y = y - offset
    h = h + 2*offset
    w = w + 2*offset

    print(w)
    print(h)
    cv2.putText(frame, "#Face", (x-10, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)       
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Show", frame)
cv2.waitKey(0)
    # if cv2.waitKey(1) == 27:
    #     break
