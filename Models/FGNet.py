from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, LocallyConnected2D, AveragePooling2D, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard

import FGNetConfig

class FGNet():
    # pass

    def __init__(self):
        self._model = Sequential()

        self._model_checkpoint = ModelCheckpoint(filepath=FGNetConfig.props['LOG_PATH'], 
                                                monitor='val_loss', 
                                                verbose=1, 
                                                save_best_only=True)
        self._tensor_board = TensorBoard(log_dir=FGNetConfig.props['LOG_PATH'], 
                                        histogram_freq=0, 
                                        write_graph=True, 
                                        write_images=True)

        self._callback_list = [self._model_checkpoint, self._tensor_board]

         # Configuration
        # LOG_PATH = './Logs'
        # MODEL_PATH = os.path.join(args.model_dir, 'weights-improvement-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5')
       


    def __reference__(self):
        self._model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu',
                        input_shape=FGNetConfig.props['INPUT_SHAPE']))
        self._model.add(AveragePooling2D(pool_size=(2, 2)))

        self._model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='relu'))
        self._model.add(AveragePooling2D(pool_size=(2, 2)))

        self._model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='valid', activation='relu'))
        self._model.add(AveragePooling2D(pool_size=(2, 2)))

        self._model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='valid', activation='relu'))
        self._model.add(AveragePooling2D(pool_size=(2, 2)))

        self._model.add(Flatten())
        self._model.add(Dense(1024, activation='relu'))
        self._model.add(Dense(6, activation='softmax'))


    def train(self, X_train, y_train, X_dev, y_dev):
        self._model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['loss'])
        self._model.fit(x=X_train, y=y_train, batch_size=FGNetConfig.props['BATCH_SIZE'], 
                                epochs=FGNetConfig.props['EPOCHS'],
                                validation_data=(X_dev, y_dev))


    def __evaluate__(self):
        pass


    