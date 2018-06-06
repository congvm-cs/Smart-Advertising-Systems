""" This script is to train our model for gender classification
    Our trained model is stored at my drive: __________________
    Download and test.
    Use --help for more informations.

    In this research:
        We proposed 2 Model descripted below.
        Accuracy on train/test subset: 1.0/0.93
        Input image size: 128 x 128
"""

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import argparse
import os
from sklearn.preprocessing import LabelEncoder  
##=============================================================================================##

IMAGE_SIZE = 64
IMAGE_DEPTH = 1

def load_images_and_labels_on_Ram(data_dir):
    print('Loading file in folder...')
    images = []
    labels = []
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        print("-> {}".format(folder_path))
        for file_name in os.listdir(folder_path):
            # lOAD IMAGES
            img = cv2.imread(os.path.join(folder_path, file_name))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            
            images.append(gray_img)

            # LOAD LABELS
            gender_label = str(file_name)[0]
            labels.append(gender_label)

    return [images, labels]


def load_images_and_labels(data_dir):
    print('Loading file in folder...')
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        print("-> {}".format(folder_path))
        for file_name in os.listdir(folder_path):

            if folder_name == 'train':
                # lOAD IMAGES
                img = cv2.imread(os.path.join(folder_path, file_name))
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            
                X_train.append(gray_img)

                # LOAD LABELS
                gender_label = str(file_name)[0]
                y_train.append(gender_label)

            if folder_name == 'test':
                # lOAD IMAGES
                img = cv2.imread(os.path.join(folder_path, file_name))
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            
                X_test.append(gray_img)

                # LOAD LABELS
                gender_label = str(file_name)[0]
                y_test.append(gender_label)
    
    return  [X_train, X_test, y_train, y_test]


# Build model
def initialize_model_2():
    # Inspired from VGG
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    print(model.summary())
    return model

def initialize_model_3():
    # Inspired from VGG
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1028, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    print(model.summary())
    return model


def load_model_1():
    print('Constructing model...')
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH)))
    model.add(BatchNormalization())
    # model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(rate=0.25))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())
    return model


def train(model, X_train, X_val, y_train, y_val, callback_list):
    print('Training...')
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x=X_train, y=y_train, batch_size=100, validation_data=(X_val, y_val), epochs=100, callbacks=callback_list)


def main(args):
    # Configuration
    LOG_PATH = './Logs'
    MODEL_PATH = os.path.join(args.model_dir, 'weights-improvement-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5')
    model_checkpoint = ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True)
    tensor_board = TensorBoard(log_dir=LOG_PATH, histogram_freq=0, write_graph=True, write_images=True)

    DATA_PATH = args.data_dir
    # OUTPUT_PATH = "C:/Users/VMC/Desktop/CAS-PEAL-R1/FRONTAL_1/"

    # Load Images and Labels
    # images, labels = load_images_and_labels(DATA_PATH)
    X_train, X_test, y_train, y_test = load_images_and_labels(DATA_PATH)
    X_train = np.reshape(X_train, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH))
    X_test = np.reshape(X_test, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH))

    # Normalize X
    X_train = X_train / 255
    X_test = X_test / 255

    # Encoding Labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)

    print('--------------------------------------------')
    print('Total images: {}'.format(X_train.shape))
    print('Total labels: {}'.format(y_train.shape))
    print('--------------------------------------------')
    
    # Split data into training and validation
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=12)
    
    # Load model 2
    if args.pretrained_model_dir == 'None':
        model = initialize_model_3()
    else:
        print('Load model from: {}'.format(args.pretrained_model_dir))
        model = load_model(args.pretrained_model_dir)

    # Callback List
    callback_list = [model_checkpoint, tensor_board]

    # Training phase
    # datagen = ImageDataGenerator(rotation_range=30,
    #                         width_shift_range=0.2,
    #                         height_shift_range=0.2,
    #                         rescale=1./255,
    #                         shear_range=0.2,
    #                         zoom_range=0.2,
    #                         horizontal_flip=False,
    #                         fill_mode='nearest')
    # datagen.
    train(model, X_train, X_test, y_train, y_test, callback_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help='data directory', type=str)
    parser.add_argument("--model_dir", help='model stored directory', type=str)
    parser.add_argument("--pretrained_model_dir", help='load pretrain model', type=str, default='None')
    args = parser.parse_args()
    main(args)