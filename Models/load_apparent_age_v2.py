import glob
import pandas as pd
import os
import sys
sys.path.append('..')

from shutil import move
from MTCNN.MTCNN import MTCNN
import cv2


def classify_into_subfolder():
    data = pd.read_csv('/home/vmc/Downloads/Untitled Folder/Reference.csv', sep=';')
    data_path = '/home/vmc/Downloads/Untitled Folder/Test'
    # print(data.values)

    for image_name, age, _ in data.values:
        # print(image_name, age)
        image_path = os.path.join(data_path, image_name)

        based_age_path = os.path.join(data_path, str(int(age)))
        if not os.path.isdir(based_age_path):
            os.mkdir(based_age_path)

        move(image_path, based_age_path)


def _split_info_by_path(image_path):
    phase_subfolder = image_path.split('/')[-3]     # Train or Test
    age_number =  image_path.split('/')[-2]         # Age
    file_name =  image_path.split('/')[-1]         # Age
    return [phase_subfolder, age_number, file_name]


def rename_by_age():
    image_path_arr = glob.glob('/home/vmc/Downloads/Apparent_Age_Dataset/*/*/*.jpg')
    based_folder_path = '/home/vmc/Downloads/Apparent_Age_Dataset'
    index = 1
    print('Number of Images: {}'.format(len(image_path_arr)))

    for image_path in image_path_arr:
        [phase_subfolder, age_number, _] = _split_info_by_path(image_path)

        new_name = '{}_{:05}.jpg'.format(age_number, index)
        new_path =  '{}/{}/{}/{}'.format(based_folder_path, phase_subfolder, age_number, new_name)

        os.rename(image_path, new_path)

        print('{} --> {}'.format(image_path, new_path))
        index += 1


def crop_face():
    # input_data_dir = ''
    output_data_dir = '/home/vmc/Downloads/Apparent_Age_Dataset_Output'
    image_size = 64
    mtcnn_model_dir = '/mnt/Data/MegaSyns/Projects/Smart-Advertising-Systems/MTCNN/Models'

    mtcnn = MTCNN(mtcnn_model_dir)

    image_path_arr = glob.glob('/home/vmc/Downloads/Apparent_Age_Dataset/*/*/*.jpg')
    # based_folder_path = '/home/vmc/Downloads/Apparent_Age_Dataset'
    
    num_success_crop = 0
    for image_path in image_path_arr:
        print('#File: {}'.format(image_path))

        [phase_subfolder, age_number, file_name] = _split_info_by_path(image_path)
        new_path =  '{}/{}/{}/'.format(output_data_dir, phase_subfolder, age_number)

        img_origin = cv2.imread(image_path)
        face_crop, ret = mtcnn.single_face_crop(img_origin)

        if not os.path.isdir(new_path):
            os.mkdir(new_path)

        if ret == False:
            continue
        else:   
            # face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            resized_face = cv2.resize(face_crop, (image_size, image_size))
            cv2.imwrite(os.path.join(new_path, file_name), resized_face)
            num_success_crop += 1

    
    print('Success rate: {}/{}'.format(len(image_path_arr), num_success_crop))   

crop_face()