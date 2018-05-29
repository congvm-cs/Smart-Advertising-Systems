import sys
sys.path.append('..')

import requests 
from cv2 import VideoCapture, resize
import cv2
import base64 
import json
import time

import config
from threading import Thread

class Client():

    def __init__(self):
        self._cap = VideoCapture(-1)
	#self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
	#self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._information = {}
        self._stop = False


    def __get_frame(self):
	print('hi')
        while not self._stop:
            self._information.clear()
            _, image = self._cap.read()
	    
            image = resize(image, (0, 0), fx=0.7, fy=0.7)	

            print(image.shape)
            image_encoded = base64.b64encode(image)
            self._information['ad_index'] = 1
            self._information['image'] = str(image_encoded)

            requests.post(config.UPLOAD_ADDRESS, json = self._information)
    

    def stop(self):
        self._stop = True


    def resume(self):
        self._stop = False

    
    def start(self):
        print('Start getting frame!')
		# start a thread to read frames from the file video stream
        t = Thread(target=self.__get_frame, args=[])
        t.daemon = True
        t.start()
