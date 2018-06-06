import sys
sys.path.append('..')
sys.path.append('./Models/')

import tensorflow as tf
import keras.utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from Models.FaceDetetion import FaceDetection
from Models import agconfig
from Models import agutils
import cv2
import numpy as np

class AGNet():

    def __init__(self, verbose=1):
        print('[LOGGING][AGNET] - Load AGModel - Loading')
        self.verbose = verbose
        self.graph = None
        self.model = self.__contruct_model()
        self.__compile()

        self.GENDER = ['Male', 'Female']
        self.AGE = ['0-18', '18-25', '25-35', '35-50', '>50']
        
        self.WEIGHT_PATH = agconfig.WEIGHT_PATH

        print('[LOGGING][AGNET] - Load AGModel - Done')

    def __contruct_model(self):
        input_x = Input((64, 64, 3))
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
        model.load_weights(agconfig.WEIGHT_PATH)

        if self.verbose:
            print(model.summary())

        self.graph = tf.get_default_graph()

        return model

    
    def __load_weights(self):
        ''' Load Pretrain weights
            WEIGHT_PATH saved at agconfig
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
        gender_sum = 0
        age_sum = [0, 0, 0, 0, 0]

        for img in images:
            face_rect_reshape = np.reshape(img, newshape=(1, agconfig.IMAGE_WIDTH, 
                                                                        agconfig.IMAGE_HEIGHT, 
                                                                        agconfig.IMAGE_DEPTH))
            face_rect_reshape = agutils.preprocess_image(face_rect_reshape)
            
            with self.graph.as_default():
                [y_gender_pred, y_age_pred] = self.model.predict(face_rect_reshape)
            
            print(y_gender_pred)
            gender_sum += y_gender_pred[-1]
            age_sum += y_age_pred
        
        # print(gender_sum/15)
        gender_pred = self.GENDER[int(np.round(gender_sum/15))]
        age_pred = self.AGE[np.argmax(y_age_pred)]

        return [gender_pred, age_pred]



