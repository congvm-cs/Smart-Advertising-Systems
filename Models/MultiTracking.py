import cv2
import dlib
import tensorflow as tf
from imutils.face_utils import rect_to_bb
import threading
import time
import numpy as np


import keras.utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import utils
import Config

class MultiTracking():

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

        #The deisred output width and height
        self.OUTPUT_SIZE_WIDTH = 775
        self.OUTPUT_SIZE_HEIGHT = 600

        self.GENDER = ['Male', 'Female']
        self.AGE = ['0-18', '18-25', '25-35', '35-60', '>60']

        self.TRACKER_STYLES = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.TRACKER_STYLE = self.TRACKER_STYLES[4]
        self.graph = None
        self.model = None

        self.faceTrackers = {}
        self.faceArr = []
        self.numEveryFaceInDict = []
        self.currentFaceID = 0          # Everyone has only separate ID number
        self.faceNames = {}


    def model_predict(self, images):
        gender_sum = 0
        age_sum = [0, 0, 0, 0, 0]

        for img in images:
            face_rect_resized = cv2.resize(img, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))

            face_rect_reshape = np.reshape(face_rect_resized, newshape=(1, Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, Config.IMAGE_DEPTH))
            face_rect_reshape = utils.preprocess_image(face_rect_reshape)

            with self.graph.as_default():
                [y_gender_pred, y_age_pred] = self.model.predict(face_rect_reshape)
            
            gender_sum += y_gender_pred[-1]
            age_sum += y_age_pred

        gender_pred = self.GENDER[int(np.round(gender_sum/15))]
        age_pred = self.AGE[np.argmax(y_age_pred)]

        return [gender_pred, age_pred]


    #We are not doing really face recognition
    def doRecognizePerson(self, faceNames, fid, images):
        print('Start predict')
        # Predict gender and age here
        # collect 10 faces to predict exactly
        [gender_pred, age_pred] = self.model_predict(images) 
        self.faceNames[fid] = "Person {}: {} {}".format(str(fid), gender_pred, age_pred)


    def contruct_model(self):
        input_x = Input((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, Config.IMAGE_DEPTH))
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_x)
        x = MaxPooling2D(strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dropout(0.2)(x)

        x = Dense(128, activation='relu')(x)

        output_gender = Dense(Config.OUTPUT_GENDER, activation='sigmoid', name='gender_output')(x)
        output_age = Dense(Config.OUTPUT_AGE, activation='softmax', name='age_output')(x)

        self.model = Model(input_x, [output_gender, output_age])
        self.model.load_weights(Config.WEIGHT_PATH)
        print(self.model.summary())
        # self.graph = tf.get_default_graph()


    def start(self):
        pass

    
    def stop(self):
        pass