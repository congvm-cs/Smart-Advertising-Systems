import tensorflow as tf
import numpy as np
from MTCNN import detect_face
import argparse
import cv2 
import logging

class MTCNN():

    def __init__(self, mtcnn_model_dir):
        # mtcnn parameters
        self._minsize = 10                      # minimum size of face
        self._threshold = [0.3, 0.4, 0.5]       # three steps's threshold
        self._factor = 0.3                      # scale factor
        self._sess = tf.Session()
        self._pnet, self._rnet, self._onet = detect_face.create_mtcnn(self._sess, mtcnn_model_dir)
        self._crop = 0
        # self._multi_crop = []

    def _to_rgb(self, img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret


    def single_face_crop(self, image):
        with tf.Graph().as_default():
            bounding_boxes, _ = detect_face.detect_face(image, 
                                                        self._minsize, 
                                                        self._pnet, 
                                                        self._rnet, 
                                                        self._onet, 
                                                        self._threshold, 
                                                        self._factor)

            nrof_faces = bounding_boxes.shape[0]    # number of faces
            # print('Number of faces: {}'.format(nrof_faces))
            
            if nrof_faces == 0:
                return self._crop, False

            for face_position in bounding_boxes:    
                face_position = face_position.astype(int)
                # cv2.rectangle(frame, (face_position[0],face_position[1]),(face_position[2],face_position[3]),(0,255,0),2)
                # Get crop image from bounding box

                # Expanding face

                x1 = face_position[1] 
                x2 = face_position[3]
                
                y1 = face_position[0]
                y2 = face_position[2]

                offset = int(0.2*(y2 - y1)) 
                x1 = x1 - offset
                y1 = y1 - offset
                y2 = y2 + offset
                x2 = x2 + offset
                self._crop = image[x1:x2, y1:y2, :]
                # Create crop image
                self._crop = cv2.cvtColor(self._crop, cv2.COLOR_BGR2RGB)

        return self._crop, True


    def multi_face_crop(self, image):
        crop_arr = []
        bounding_boxes = []
        with tf.Graph().as_default():
            bounding_boxes, _ = detect_face.detect_face(image, 
                                                        self._minsize, 
                                                        self._pnet, 
                                                        self._rnet, 
                                                        self._onet, 
                                                        self._threshold, 
                                                        self._factor)

            nrof_faces = bounding_boxes.shape[0]    # number of faces
            # print('Number of faces: {}'.format(nrof_faces))
            
            if nrof_faces == 0:
                return crop_arr, bounding_boxes, False

            for face_position in bounding_boxes:    
                face_position = face_position.astype(int)
                # cv2.rectangle(frame, (face_position[0],face_position[1]),(face_position[2],face_position[3]),(0,255,0),2)
                # Get crop image from bounding box

                x1 = face_position[1] 
                x2 = face_position[3]
                
                y1 = face_position[0]
                y2 = face_position[2]

                offset = int(0.2*(y2 - y1)) 
                # offset = 0
                x1 = x1 - offset
                y1 = y1 - offset
                y2 = y2 + offset
                x2 = x2 + offset
                crop = image[x1:x2, y1:y2, :]
                # Create crop image
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_arr.append(crop)
                
        return [crop_arr, bounding_boxes, True]