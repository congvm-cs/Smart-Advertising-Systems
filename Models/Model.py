import keras.utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from FaceDetetion import FaceDetection
import agconfig
import agutils
import cv2
import numpy as np


class AGNet():

    def __init__(self, input_shape, verbose=1):
        self.input_shape = input_shape
        self.verbose = verbose

        self.model = self.__model()
        self.__compile()
        self.detector = FaceDetection()

        self.GENDER = ['Male', 'Female']
        self.AGE = ['0-18', '18-25', '25-35', '35-60', '>60']


    def __model(self):
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

        x = Flatten()(x)
        x = BatchNormalization()(x)

        x = Dense(128, activation='relu')(x)

        output_gender = Dense(1, activation='sigmoid', name='gender_output')(x)
        output_age = Dense(6, activation='softmax', name='age_output')(x)

        model = Model(input_x, [output_gender, output_age])

        if self.verbose:
            print(model.summary)

        return model


    def __compile(self):
        self.model.compile( optimizer='Adam', 
                            loss={'gender_output': 'binary_crossentropy',
                                  'age_output': 'categorical_crossentropy'}, 
                            metrics={'gender_output': 'accuracy',
                                     'age_output': 'accuracy'},
                            loss_weights={'gender_output': 1.0,
                                          'age_output': 1.0})


    def predict(self, image):
        gender_arr = []
        age_arr = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector.detectMultiFaces(gray)

        for face in faces:
            (x, y, w, h) = face

            crop_image = image[y:y+h, x:x+w, :]
            img = agutils.preprocess_image_for_test(crop_image)
            [y_gender_pred, y_age_pred] =  self.model.predict(img)

            gender_pred = self.GENDER[y_gender_pred]
            age_pred = self.AGE[np.argmax(y_age_pred)]        

            gender_arr.append(gender_pred)
            age_arr.append(age_pred)

        # Count number of males, females and ages of them

