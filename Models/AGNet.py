from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, AveragePooling2D, Flatten, BatchNormalization, ZeroPadding2D, Convolution2D, Merge
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
import AGNetConfig
import os
from keras.applications.vgg16 import VGG16
from keras import Model

class AGNet():
    # pass

    def __init__(self):
        if not os.path.isdir(AGNetConfig.props['MODEL_PATH']):
            os.mkdir(AGNetConfig.props['MODEL_PATH'])
        
        if not os.path.isdir(AGNetConfig.props['LOG_PATH']):
            os.mkdir(AGNetConfig.props['LOG_PATH'])
        
        model_checkpoint_name = os.path.join(AGNetConfig.props['MODEL_PATH'], AGNetConfig.props['WEIGHT_NAME'])

        self._model_checkpoint = ModelCheckpoint(filepath=model_checkpoint_name, 
                                                monitor='val_loss', 
                                                verbose=1, 
                                                save_best_only=True)
        self._tensor_board = TensorBoard(log_dir=AGNetConfig.props['LOG_PATH'], 
                                        histogram_freq=0, 
                                        write_graph=True, 
                                        write_images=True)
        self._callback_list = [self._model_checkpoint, self._tensor_board]


    def __reference__(self):
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                        input_shape=AGNetConfig.props['INPUT_SHAPE']))
        model.add(Dropout(0.2))
        model.add(AveragePooling2D(pool_size=(2, 2)))                        
        
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))   
        model.add(Dropout(0.2))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.2))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.2))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.2))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(filters=2048, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Dropout(0.2))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(6, activation='sigmoid'))

        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        print(model.summary())
        return model


    def __vgg16_model__(self):
        """VGG 16 Model 
        Parameters:
        img_rows, img_cols - resolution of inputs
        channel - 1 for grayscale, 3 for color 
        num_classes - number of categories for our classification task
        """
        (channel, img_rows, img_cols) = (3, 64, 64)
        num_classes = 7

        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Loads ImageNet pre-trained data
        model.load_weights('/home/vmc/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

        # Truncate and replace softmax layer for transfer learning
        # Add Fully Connected Layer
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        # model.add(Dense(4096, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(1000, activation='softmax'))
        
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.add(Dense(num_classes, activation='softmax'))

        print(model.summary())

        # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
        for layer in model.layers[:10]:
            layer.trainable = False

        # Learning rate is changed to 0.001
        # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model


    def train(self, X_train, y_train, X_dev, y_dev):
        self._model = self.__reference__()
        # self._model = self.__vgg16_model__()
        self._model.fit(x=X_train, y=y_train, batch_size=AGNetConfig.props['BATCH_SIZE'], 
                                epochs=AGNetConfig.props['EPOCHS'],
                                validation_data=(X_dev, y_dev),
                                callbacks=self._callback_list)


    def __evaluate__(self):
        pass


    