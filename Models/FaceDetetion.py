import agconfig
import cv2
import dlib
from imutils.face_utils import rect_to_bb

class FaceDetection():
    ''' Face Detection is a combined class that contains 2 face detector: 
        dlib face detector and haar detector

        Chosen detector is set in Config.py with DETECTION_METHOD parameter.
        2 multi-face detector availabel now are HAAR and DLIB
    '''
    def __init__(self):
        print('[LOGGING][FACE DETECTION] - Load Face Detection - Loading')
        self.method = agconfig.DETECTION_METHOD
        self.detector = self.__get_detector()
        print('[LOGGING][FACE DETECTION] - Load Face Detection - Done')


    def __get_detector(self):
        if agconfig.DETECTION_METHOD == 'DLIB':
            return dlib.get_frontal_face_detector()

        elif agconfig.DETECTION_METHOD == 'HAAR':
            return cv2.CascadeClassifier(agconfig.HAAR_MODEL_PATH)

    
    def detectMultiFaces(self, gray):
        if self.method == 'DLIB':
            faces = self.detector(gray, 1)    
            faces = map(rect_to_bb, faces)

        elif self.method == 'HAAR':
            faces = self.detector.detectMultiScale(gray, 1.3, 6)  
        return faces