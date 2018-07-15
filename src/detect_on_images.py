# system modules
import sys
sys.path.append('..')

import cv2
import time
import numpy as np
import threading
# local modules
from src.config import config
from src.ultis import utils
from src import FaceDetetion
from src.models import AGNet
from src import Person

agModel = AGNet.AGNet(verbose=False)
detector = FaceDetetion.FaceDetection()   
rectangleColor = (0, 255, 0)
current_id = 0

baseImage = cv2.imread('D:/Dataset/Face_Data/The_Images_of_Groups_Dataset/Fam4a/2597052300_11e43ea94b_3099_45695345@N00.jpg')
gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiFaces(gray)


for face in faces:
    (x, y, w, h) = face
    offset = int(0.1*w)
    # offset = 0
    t_x = x - offset 
    t_y = y - offset
    t_w = x + w + offset 
    t_h = y + h + offset

    person = Person.Person(current_id)

    utils.draw_rectangle(baseImage, t_x, t_y, t_w, t_h, rectangleColor)
    # utils.draw_rectangle(baseImage, x, y, x + w, y + h, rectangleColor)
    crop_image = baseImage[t_y : t_h, t_x : t_w, :]
    # cv2.imshow('t', crop_image)
    # cv2.waitKey(0)
    crop_image_resized = cv2.resize(crop_image, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    crop_image_resized = np.reshape(crop_image_resized, newshape=(1, config.IMAGE_WIDTH, 
                                                                        config.IMAGE_HEIGHT, 
                                                                        config.IMAGE_DEPTH))
    print(crop_image_resized.shape)
    [gender_pred, age_pred] = agModel.predict_with_array(crop_image_resized)

    cv2.putText(baseImage, "Person: " + str(person.getId()), (int(t_x), int(t_y - t_h/4)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


    cv2.putText(baseImage, "#: " + str(gender_pred) , 
                (int(t_x), int(t_y - t_h/7)), 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)


    cv2.putText(baseImage, "#: " + str(age_pred) , 
                (int(t_x), int(t_y - t_h/19)), 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # cv2.putText(baseImage, "#: " + str(gender_pred) , 
    #             (int(t_x), int(t_y - t_h/5)), 
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             t_w/400, (0, 255, 255), 2)


    # cv2.putText(baseImage, "#: " + str(age_pred) , 
    #             (int(t_x), int(t_y - t_h/19)), 
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             t_w/400, (0, 255, 255), 2)
    
    current_id += 1

cv2.imshow('Result', baseImage)
cv2.waitKey(0)