# system modules
import sys
sys.path.append('..')

import tensorflow as tf
import keras.utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
import cv2
import numpy as np


# local modules
from src.FaceDetetion import FaceDetection
from src.config import config
from src.ultis import utils


class AGNet():
    def __init__(self, verbose=1):
        """ AGNet for gender and age classification on facial images
            Parameter:
                verbose :   show detailed model on console
                graph   :   tensorflow graph
                model   :   keras model instance 
        """
        print('[LOGGING][AGNET] - Load AGModel - Loading')
        self.verbose = verbose
        self.graph = None
        self.model = self.__contruct_model()
        self.__compile()

        self.GENDER = ['Male', 'Female']
        self.AGE = ['0-12', '12-18', '18-25', '25-35', '35-50', '>50']
        
        self.WEIGHT_PATH = config.WEIGHT_PATH

        print('[LOGGING][AGNET] - Load AGModel - Done')


    def __contruct_model(self):
        input_x = Input((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_DEPTH))
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

        output_gender = Dense(1, activation='sigmoid', name='gender_output')(x)
        output_age = Dense(5, activation='softmax', name='age_output')(x)

        model = Model(input_x, [output_gender, output_age])
        model.load_weights(config.WEIGHT_PATH)
        # input_x = Input((64, 64, 3))
        # x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_x)
        # x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_x)
        # x = MaxPooling2D(strides=(2, 2))(x)
        # x = BatchNormalization()(x)

        # x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        # x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        # x = MaxPooling2D(strides=(2, 2))(x)
        # x = BatchNormalization()(x)

        # x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
        # x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
        # x = MaxPooling2D(strides=(2, 2))(x)
        # x = BatchNormalization()(x)

        # x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
        # x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
        # x = MaxPooling2D(strides=(2, 2))(x)
        # x = BatchNormalization()(x)

        # x = Flatten()(x)
        # x = Dropout(0.2)(x)

        # x = Dense(128, activation='relu', name='embedded_layer')(x)

        # output_gender = Dense(1, activation='sigmoid', name='gender_output')(x)
        # output_age = Dense(6, activation='softmax', name='age_output')(x)

        # model = Model(input_x, [output_gender, output_age])
        # model.load_weights(config.WEIGHT_PATH)
        # model.load_weights('./model.h5')
        # print(model.summary())
        if self.verbose:
            print(model.summary())

        self.graph = tf.get_default_graph()

        return model

    
    def __load_weights(self):
        ''' Load Pretrain weights
            WEIGHT_PATH saved at config
        '''
        print('[LOGGING][AGNet] - Load weights - Loadding')
        self.model.load_weights(self.WEIGHT_PATH)
        print('[LOGGING][AGNet] - Load weights - Done')


    def __compile(self):
        self.model.compile( optimizer='Adam', 
                            loss={'gender_output': 'binary_crossentropy',
                                  'age_output': 'categorical_crossentropy'}, 
                            metrics={'gender_output': 'accuracy',
                                     'age_output': 'accuracy'},
                            loss_weights={'gender_output': 1.0,
                                          'age_output': 1.0})


    def predict_with_array(self, images):
        """ Predict using set of images collected from camera
            In this case, we use 10 cropped face to predict gender and age,
            then using average predicted number to confirm.

            Parameters:
                image   :   nparray, list or tuple
            
            Return:
                 [gender_pred, age_pred]
        """
        gender_sum = 0
        age_sum = [0, 0, 0, 0, 0]

        div = len(images)

        for img in images:
            face_rect_reshape = np.reshape(img, newshape=(1, config.IMAGE_WIDTH, 
                                                                        config.IMAGE_HEIGHT, 
                                                                        config.IMAGE_DEPTH))
            face_rect_reshape = utils.preprocess_image(face_rect_reshape)
            
            with self.graph.as_default():
                [y_gender_pred, y_age_pred] = self.model.predict(face_rect_reshape)
            
            print(y_gender_pred)
            gender_sum += y_gender_pred[-1]
            age_sum += y_age_pred
        
        # print(gender_sum/15)
        gender_pred = self.GENDER[int(np.round(gender_sum/div))]
        age_pred = self.AGE[np.argmax(y_age_pred)]

        return [gender_pred, age_pred]



