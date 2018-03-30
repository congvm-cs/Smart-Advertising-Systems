import cv2
import os
# import Model Class

import random


def categorize_labels(gender, age):
    labels = [0, 0, 0, 0, 0, 0, 0]
    age = int(age)

    if gender == '1':
        labels[0] = 1   # Female
    else:
        labels[0] = 0   # Male

    if 0 < age and age < 12:
        labels[1] = 1
    elif 13 < age and age < 19:
        labels[2] = 1
    elif 20 < age and age < 36:
        labels[3] = 1
    elif 37 < age and age < 65:
        labels[4] = 1
    elif age > 66:
        labels[5] = 1

    return labels


def load_images_and_labels(data_dir, load_image_path = True):
    images = []
    labels = []

    for folder_name in os.listdir(data_dir):
        if folder_name[-3:] == 'txt':
                # print(file_name)
            continue
            
        folder_path = os.path.join(data_dir, folder_name)

        for file_name in os.listdir(folder_path):
            
            file_path = os.path.join(folder_path, file_name)

            # Append images
            if load_image_path == True:
                images.append(file_path)
            else:
                I = cv2.imread(file_path)
                images.append(I)
                
            # File name: A1-G1-0-1025296488_4712c26a4f_1160_96603368@N00-Fam2a.jpg
            # Append label
            num_age = file_name.split('-')[0][1:]
            gender = file_name.split('-')[1][1:]
            label = categorize_labels(gender, num_age)
            labels.append(label)

    return [images, labels]

def train():
    pass


def test():
    data_dir = '/media/vmc/12D37C49724FE954/Face_Data/The_Images_of_Groups_Dataset/GA_Net_database'
    images, labels = load_images_and_labels(data_dir)

    for _ in range(4):
        index = random.randint(0, 10000)       
        print(images[index])
        print(labels[index])
        print('--------------------------------------')

if __name__ == '__main__':
    test()