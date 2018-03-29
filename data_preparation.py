"""Usage:
    folder structure:
        dataset:
            -> subfolder_1
            -> subfolder_2
            ...
        
        processed:
            -> subfolder_1
            -> subfolder_2
            ...
        
        final:
            -> train_folder
            -> test_folder
"""

import os
import cv2
import argparse
from sys import stdout, stdin
from shutil import copy2, move
import numpy as np
import dlib
from imutils.face_utils import rect_to_bb, FaceAligner


class Data_Preparation():

    def __init__(self, data_dir, processed_data_dir, final_dir, crop_size):
        self._data_dir = data_dir
        self._processed_data_dir = processed_data_dir
        self._final_dir = final_dir
        self._crop_size = crop_size

        self._train_dir = os.path.join(self._final_dir, 'train') 
        self._test_dir = os.path.join(self._final_dir, 'test')


    def split_and_save_data(self):
        if not os.path.isdir(self._train_dir):        
            os.mkdir(self._train_dir)
        
        if not os.path.isdir(self._test_dir):
            os.mkdir(self._test_dir)

        print('Spliting file in folder...')
        images_path = []

        for folder_name in os.listdir(self._processed_data_dir):
            folder_path = os.path.join(self._processed_data_dir, folder_name)
            print("-> {}".format(folder_path))
            for file_name in os.listdir(folder_path):        
                # lOAD IMAGES
                images_path.append(os.path.join(folder_path, file_name))

        # print(len(images_path))
        print('{} files in {}'.format(len(images_path), self._processed_data_dir))

        # Shuffle data
        np.random.shuffle(images_path)

        # Split data into train and test
        train_path = images_path[0:int(0.9*len(images_path))]
        test_path = images_path[int(0.9*len(images_path))::]

        for index, _file in enumerate(train_path):
            move(_file, self._train_dir)
            print('copying training file  ... | {:.2f} %'.format(index*100/len(train_path)), flush=True, end='\r')

        for index, _file in enumerate(test_path):
            move(_file, self._test_dir)
            print('copying test file... | {:.2f} %'.format(index*100/len(test_path)), flush=True, end='\r')


    def load_and_crop_face(self):

        print('Loading and crop face...')

        detector = dlib.get_frontal_face_detector()

        for folder_name in os.listdir(self._data_dir):

            processed_folder_path = os.path.join(self._processed_data_dir, folder_name)
            if not os.path.isdir(processed_folder_path):
                os.mkdir(processed_folder_path)

            folder_path = os.path.join(self._data_dir, folder_name)
            print("-> {}".format(folder_path))

            for file_name in os.listdir(folder_path):        
                file_path = os.path.join(folder_path, file_name)
                print('{} - {}'.format('load_and_crop_face', file_path))

                # lOAD IMAGE
                    
                I_data = cv2.imread(os.path.join(folder_path, file_name))
                I_gray = cv2.cvtColor(I_data, cv2.COLOR_BGR2GRAY)

                rects = detector(I_gray, 1)

                for rect in rects:
                    (x, y, w, h) = rect_to_bb(rect)             # Positions of rectangle contains face\

                    # Expanding face
                    offset = int(0.25*x) 
                    x = x - offset
                    y = y - offset
                    h = h + 2*offset
                    w = w + 2*offset

                    I_face = I_gray[y:y+w, x:x+h]
                    resized_face = cv2.resize(I_face, (self._crop_size, self._crop_size))
                    cv2.imwrite(os.path.join(processed_folder_path, file_name), resized_face)


def main(args):
    print('Data Preparation ...')
    preparation = Data_Preparation(args.data_dir, args.processed_data_dir, args.final_dir, args.crop_size)
    preparation.load_and_crop_face()
    preparation.split_and_save_data()
    print('Done!')


def test(image, flag):
    cv2.imshow(flag, image)
    cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help='data directory', type=str)
    parser.add_argument("--processed_data_dir", help='processed folder directory', type=str)
    parser.add_argument("--final_dir", help='final directory', type=str)
    # parser.add_argument("--train_dir", help='training directory', type=str)
    # parser.add_argument("--test_dir", help='test', type=str)
    parser.add_argument("--crop_size", help='image size', type=int, default=128)
    args = parser.parse_args()
    main(args)