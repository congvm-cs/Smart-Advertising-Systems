import requests
import numpy
import cv2
import skvideo.io

vid = skvideo.io.vreader('/Users/ngocphu/Documents/FINAL_PROJECT_RESEARCH/Smart_Advertising_Systems/Smart-Advertising-Systems/flask_streaming/1.avi')
for frame in vid:
    cv2.imshow('asdas', frame)
    cv2.waitKey(1)
cv2.release()
cv2.destroyAllWindows