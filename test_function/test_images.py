import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import tensorflow as tf

import os
import keras.utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import dlib
from imutils.face_utils import rect_to_bb


def contruct_model():
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
    print(model.summary())

    return model


def preprocess_image(img):   
    def __prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y 

    img = img/255.0
    img = __prewhiten(img)
    return img


print('Load model...')
# model_path = '/mnt/e/Smart-Advertising-Systems-master/Models/weights-improvement-23-0.23-0.92.hdf5'
model_path = '/home/vmc/Downloads/AGNet_weights_1-improvement-30-0.22-0.90.hdf5'
align_predictor_path = '/mnt/Data/MegaSyns/Projects/Smart-Advertising-Systems/dlib_face_landmarks_model/shape_predictor_68_face_landmarks.dat'

# Camera Streaming
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
model = contruct_model()
model.load_weights('/home/vmc/Downloads/train-weights-model-lastest(2).h5')

face_id = []

index = 0
y_gen = 0
gender = ''

image_paths = glob.glob('/home/vmc/Desktop/*.jpg')

print(len(image_paths))

for p in image_paths:
    frame = cv2.imread(p)
    image = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rects = detector(gray, 1)

    img_clone = image.copy()

    for (i, rect) in enumerate(rects):
        (x1, y1, w, h) = rect_to_bb(rect)             # Positions of rectangle contains face
        x2 = x1 + h
        y2 = y1 + w

        # offset = int(0.15*w) 
        offset = 0
        x1 = (x1 - offset) if (x1 - offset) > 0 else 0 
        y1 = (y1 - offset) if (y1 - offset) > 0 else 0
        y2 = (y2 + offset) if (y2 + offset) < image.shape[1] else image.shape[1]
        x2 = x2 + offset if (x2 + offset) < image.shape[0] else image.shape[0]
        crop_im = image[y1:y2, x1:x2, :]

        cv2.imshow('detect', crop_im)
        cv2.waitKey(0)
        face_rect_resized = cv2.resize(crop_im, (64, 64))
        
        # cv2.imshow('after aligned #{}'.format(str(i)), face_rect_resized)
        face_rect_reshape = np.reshape(face_rect_resized, newshape=(1, 64, 64, 3))

        face_rect_reshape = preprocess_image(face_rect_reshape)

        # cv2.imshow('face crop 1', face_rect_reshape)

        [y_gender_pred, y_age_pred] = model.predict(face_rect_reshape)

    
        y_gen = y_gender_pred[-1]

        if y_gen < 0.5:
            gender = 'Nam'
        else: 
            gender = 'Nu'

        print(y_age_pred)
        age_pred = np.argmax(y_age_pred)

        cv2.putText(img_clone, "#{} #{} #{:2f}\n".format(i+1, gender, float(age_pred)), (x1-10, y1-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)       
        
        
        cv2.rectangle(img_clone, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('detect', img_clone)
    cv2.waitKey(0)