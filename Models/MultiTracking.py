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


class MultiTracking():

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

        #The deisred output width and height
        self.OUTPUT_SIZE_WIDTH = 775
        self.OUTPUT_SIZE_HEIGHT = 600
        self.WEIGHT_PATH = '/home/vmc/Downloads/train-weights-model-lastest(2).h5'

        self.GENDER = ['Male', 'Female']
        self.AGE = ['0-18', '18-25', '25-35', '35-60', '>60']

        self.TRACKER_STYLES = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.TRACKER_STYLE = self.TRACKER_STYLES[4]
        self.graph = None
        self.model = None

def saturation(self, val, min_val, max_val):
    if val > max_val:
        val = max_val
    elif val < min_val:
        val = min_val

    return val

def preprocess_image(self, img):  
    def __prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y 

    img = img/255.0
    img = __prewhiten(img)
    return img


def model_predict(self, images):
    gender_sum = 0
    age_sum = [0, 0, 0, 0, 0]

    for img in images:
        face_rect_resized = cv2.resize(img, (64, 64))

        face_rect_reshape = np.reshape(face_rect_resized, newshape=(1, 64, 64, 3))
        face_rect_reshape = self.preprocess_image(face_rect_reshape)

        global graph
        with graph.as_default():
            [y_gender_pred, y_age_pred] = self.model.predict(face_rect_reshape)
        
        # print(y_gender_pred)
        gender_sum += y_gender_pred[-1]
        age_sum += y_age_pred

    gender_pred = GENDER[int(np.round(gender_sum/15))]
    age_pred = AGE[np.argmax(y_age_pred)]

    return [gender_pred, age_pred]


#We are not doing really face recognition
def doRecognizePerson(self, faceNames, fid, images):
    # time.sleep(2
    print('Start predict')
    # Predict gender and age here

    # collect 10 faces to predict exactly

    # face_arr.append()
    # if len(face_arr) == 10:
    # while True:
    #     pass

    [gender_pred, age_pred] = self.model_predict(images)
        
    faceNames[fid] = "Person {}: {} {}".format(str(fid), gender_pred, age_pred)


def contruct_model(self):
    # graph
    input_x = Input((64, 64, 3))
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)

    x = Dense(128, activation='relu')(x)

    output_gender = Dense(1, activation='sigmoid', name='gender_output')(x)
    output_age = Dense(5, activation='softmax', name='age_output')(x)

    model = Model(input_x, [output_gender, output_age])
    model.load_weights(WEIGHT_PATH)
    print(model.summary())

    self.graph = tf.get_default_graph()

    return model


def draw_rectangle(self, img, p1, p2, p3, p4, color):
    offset = int((p3 - p1)/4)
    thickness_heavy_line = 3
    thickness_slim_line = 1

    # Left Top (p1, p2)
    cv2.line(img, (p1, p2), (p1, p2 + offset), color, thickness_heavy_line)
    cv2.line(img, (p1, p2), (p1 + offset, p2 ), color, thickness_heavy_line)
    
    # Left Bottom (p1, p4)
    cv2.line(img, (p1, p4), (p1, p4 - offset), color, thickness_heavy_line)
    cv2.line(img, (p1, p4), (p1 + offset, p4 ), color, thickness_heavy_line)

    # Right Top (p3, p2)
    cv2.line(img, (p3, p2), (p3, p2 + offset), color, thickness_heavy_line)
    cv2.line(img, (p3, p2), (p3 - offset, p2), color, thickness_heavy_line)

    # Right Bottom (p3, p4)
    cv2.line(img, (p3, p4), (p3, p4 - offset), color, thickness_heavy_line)
    cv2.line(img, (p3, p4), (p3 - offset, p4 ), color, thickness_heavy_line)
    
    cv2.line(img, (p1, p2), (p1, p4), color, thickness_slim_line)
    cv2.line(img, (p1, p2), (p3, p2), color, thickness_slim_line)
    cv2.line(img, (p3, p4), (p1, p4), color, thickness_slim_line)
    cv2.line(img, (p3, p4), (p3, p2), color, thickness_slim_line)
    return img


