import keras.utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K


class CPNet():

    def __init__(self, input_shape, verbose=1):
        self.input_shape = input_shape
        self.verbose = verbose

        self.model = self.__model()
        self.__compile()


    def __model(self):
        input_x = Input((64, 64, 3))
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_x)
        # x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_x)
        x = MaxPooling2D(strides=(2, 2))(x)
        x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        # x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(strides=(2, 2))(x)
        x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
        # x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(strides=(2, 2))(x)
        x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)

        x = Flatten()(x)
        # x = Dropout(0.2)(x)
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

    