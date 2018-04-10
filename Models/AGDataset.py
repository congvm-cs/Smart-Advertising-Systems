# from MTCNN import MTCNN
import argparse
import os
import sys
sys.path.append('..')
import cv2
from shutil import move
from sklearn.model_selection import train_test_split
import numpy as np
import AGNetConfig
from MTCNN import MTCNN
from keras.preprocessing.image import ImageDataGenerator
class AGDataset():

    def __init__(self):
        self._IMAGE_SIZE = AGNetConfig.props['IMAGE_SIZE']
        self._IMAGE_DEPTH = AGNetConfig.props['IMAGE_DEPTH']


    def load_dataset(self, args):
        print('Load data..')
        train_dir = args.train_dir
        test_dir = args.test_dir

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        for file_name in os.listdir(train_dir):
            file_path = os.path.join(train_dir, file_name)
            
            origin_I_train = cv2.imread(str(file_path))

            if int(self._IMAGE_DEPTH) == 1:
                origin_I_train = cv2.cvtColor(origin_I_train, cv2.COLOR_BGR2GRAY)

            X_train.append(origin_I_train)
            y_train.append(self.categorize_labels(file_name))

        for file_name in os.listdir(test_dir):
            file_path = os.path.join(test_dir, file_name)
            
            origin_I_train = cv2.imread(str(file_path))
            
            if int(self._IMAGE_DEPTH) == 1:
                origin_I_train = cv2.cvtColor(origin_I_train, cv2.COLOR_BGR2GRAY)

            X_test.append(origin_I_train)
            y_test.append(self.categorize_labels(file_name))

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        X_train = np.reshape(X_train, newshape=(len(X_train), self._IMAGE_SIZE, self._IMAGE_SIZE, self._IMAGE_DEPTH))
        X_test = np.reshape(X_test, newshape=(len(X_test), self._IMAGE_SIZE, self._IMAGE_SIZE, self._IMAGE_DEPTH))


        #Normalize
        X_train = X_train/255.0
        X_test = X_test/255.0
        
        return [X_train, X_test, y_train, y_test]   
        

    def data_augment(self, args):
        train_dir = args.train_dir
        test_dir = args.test_dir
        
        datagen = ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                # rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

        for file_name in os.listdir(train_dir):
            file_path = os.path.join(train_dir, file_name)
            
            origin_I_train = cv2.imread(str(file_path))
            origin_I_train = cv2.cvtColor(origin_I_train, cv2.COLOR_BGR2RGB)
            origin_I_train = origin_I_train.reshape((1,) + origin_I_train.shape)  # this is a Numpy array with shape (1, 3, 150, 150


            if int(self._IMAGE_DEPTH) == 1:
                origin_I_train = cv2.cvtColor(origin_I_train, cv2.COLOR_BGR2GRAY)

            if num_age == str(26):
                i = 0
                for batch in datagen.flow(origin_I_train, batch_size=1,
                                    save_to_dir=train_dir, save_prefix=file_name, save_format='jpeg'):
                    i += 1
                    if i > 3:
                        break  # otherwise the generator would loop indefinitely
            else:
                i = 0
                for batch in datagen.flow(origin_I_train, batch_size=1,
                                    save_to_dir=train_dir, save_prefix=file_name, save_format='jpeg'):
                    i += 1
                    if i > 4:
                        break  # otherwise the generator would loop indefinitely


        for file_name in os.listdir(test_dir):
            file_path = os.path.join(test_dir, file_name)
            
            origin_I_train = cv2.imread(str(file_path))
            origin_I_train = cv2.cvtColor(origin_I_train, cv2.COLOR_BGR2RGB)
            origin_I_train = origin_I_train.reshape((1,) + origin_I_train.shape)  # this is a Numpy array with shape (1, 3, 150, 150

            if int(self._IMAGE_DEPTH) == 1:
                origin_I_train = cv2.cvtColor(origin_I_train, cv2.COLOR_BGR2GRAY)
            num_age = file_name.split('_')[0]

            if num_age == str(26):
                i = 0
                for batch in datagen.flow(origin_I_train, batch_size=1,
                                    save_to_dir=train_dir, save_prefix=file_name, save_format='jpeg'):
                    i += 1
                    if i > 3:
                        break  # otherwise the generator would loop indefinitely
            else:
                i = 0
                for batch in datagen.flow(origin_I_train, batch_size=1,
                                    save_to_dir=train_dir, save_prefix=file_name, save_format='jpeg'):
                    i += 1
                    if i > 4:
                        break  # otherwise the generator would loop indefinitely


    def categorize_labels(self, file_name):
        # File name: A1-G1-0-1025296488_4712c26a4f_1160_96603368@N00-Fam2a.jpg
        num_age = file_name.split('_')[0]
        gender = file_name.split('_')[1]

        labels = [0, 0, 0, 0, 0, 0]
        # labels = [0, 0, 0, 0, 0, 0]
        age = int(num_age)

        if gender == '1':
            labels[0] = 1   # Female
        else:
            labels[0] = 0   # Male

        if 0 <= age and age <= 18:
            labels[1] = 1
        elif 18 < age and age <= 25:
            labels[2] = 1
        elif 25 < age and age <= 35:
            labels[3] = 1
        elif 36 < age and age <= 50:
            labels[4] = 1
        elif age > 50:
            labels[5] = 1
        return labels


    def split_train_test(self, args):
        data_dir = args.input_data_dir
        train_dir = args.train_dir
        test_dir = args.test_dir
        
        file_paths_arr = []
        # labels_arr = []

        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            file_paths_arr.append(file_path)

        print('Number of images: {}'.format(len(file_paths_arr)))

        [file_paths_train, file_paths_test] = train_test_split(file_paths_arr, random_state=10, test_size=0.05)

        if not os.path.isdir(train_dir):
            os.mkdir(train_dir)

        if not os.path.isdir(test_dir):
            os.mkdir(test_dir)  

        for index, file_path in enumerate(file_paths_train):
            move(file_path, train_dir)
            print('copying training file  ... | {:.2f} %'.format(index*100/len(file_paths_train)), flush=True, end='\r')

        for index, file_path in enumerate(file_paths_test):
            move(file_path, test_dir)
            print('copying test file... | {:.2f} %'.format(index*100/len(file_paths_test)), flush=True, end='\r')


    def crop_face_from_image(self, args):
        input_data_dir = args.input_data_dir
        output_data_dir = args.output_data_dir
        image_size = args.image_size
        mtcnn_model_dir = args.mtcnn_model_dir

        mtcnn = MTCNN.MTCNN(mtcnn_model_dir)
        
        print('Loading image from: {}'.format(input_data_dir))

        for folder_name in os.listdir(input_data_dir):
            folder_path = os.path.join(input_data_dir, folder_name)
            print('>>> Loading image from folder: {}'.format(folder_name))
            
            for file_name in os.listdir(folder_path):
                # print(str(file_name).split('.')[1])
                
                if str(file_name)[-3:] == 'JPG':
                    file_path = os.path.join(folder_path, file_name)

                    print('#File: {}'.format(file_path))

                    img_origin = cv2.imread(str(file_path))
                    face_crop, ret = mtcnn.single_face_crop(img_origin)
                
                    if ret == False:
                        continue
                    else:   
                        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
                        resized_face = cv2.resize(face_crop, (image_size, image_size))
                        cv2.imwrite(os.path.join(output_data_dir, file_name), resized_face)


def main(args):
    fgdata = AGDataset()
    # fgdata.crop_face_from_image(args)
    # fgdata.split_train_test(args)
    fgdata.data_augment(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mtcnn_model_dir', help='mtcnn model directory', default=None, type=str)
    parser.add_argument('--input_data_dir', help='input data directory', default=None, type=str)
    parser.add_argument('--output_data_dir', help='output data directory', default=None, type=str)
    parser.add_argument('--train_dir', help='train data directory', default=None, type=str)
    parser.add_argument('--test_dir', help='test data directory', default=None, type=str)
    
    parser.add_argument('--image_size', help='image size', default=64, type=int)
    args = parser.parse_args()
    main(args)
