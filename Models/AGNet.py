from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, LocallyConnected2D, AveragePooling2D, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard
import AGNetConfig
import os

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
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu',
                        input_shape=AGNetConfig.props['INPUT_SHAPE']))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='relu'))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='valid', activation='relu'))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='valid', activation='relu'))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(7, activation='softmax'))
        return model


    def train(self, X_train, y_train, X_dev, y_dev):
        self._model = self.__reference__()
        self._model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self._model.fit(x=X_train, y=y_train, batch_size=AGNetConfig.props['BATCH_SIZE'], 
                                epochs=AGNetConfig.props['EPOCHS'],
                                validation_data=(X_dev, y_dev),
                                callbacks=self._callback_list)


    def __evaluate__(self):
        pass


    