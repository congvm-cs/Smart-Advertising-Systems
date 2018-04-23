# import dlib
import numpy as np 
# from keras.models import load_model
# import cv2
# from imutils.face_utils import rect_to_bb, FaceAligner
from os import rename
import os
from shutil import copy, move
import glob

# print('Load model...')
# # model_path = '/mnt/e/Smart-Advertising-Systems-master/Models/weights-improvement-23-0.23-0.92.hdf5'
# model_path = '/home/vmc/Downloads/AGNet_weights_1-improvement-30-0.22-0.90.hdf5'
# align_predictor_path = '/mnt/Data/MegaSyns/Projects/Smart-Advertising-Systems/dlib_face_landmarks_model/shape_predictor_68_face_landmarks.dat'

# detector = dlib.get_frontal_face_detector()
# model = load_model(model_path)
# predictor = dlib.shape_predictor(align_predictor_path)

output_folder = '/mnt/Data/Dataset/Face_Data/Output'
def load_data(direcs):
    paths = []
    for file_name in os.listdir(direcs):
        file_path = os.path.join(direcs, file_name)
        paths.append(file_path)
    return paths

# new_folder = '/home/vmc/Downloads/Output_CACD2000/'
image_paths = load_data('/mnt/Data/Dataset/Face_Data/Output')
print('No. Images: {}'.format(len(image_paths)))

# image_paths = glob.glob('/mnt/Data/Dataset/Face_Data/Output/*/*.jpg')

for image_path in image_paths:
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # rects = detector(gray, 1)

    # for (i, rect) in enumerate(rects):
    #     (x, y, h, w) = rect_to_bb(rect)             # Positions of rectangle contains face

    #     offset = int(0.2*x) 
    #     x = (x - offset) if (x - offset) > 0 else 0
    #     y = y - offset   if (y - offset) > 0 else 0
    #     h = h + 2*offset if (h + 2*offset) < image.shape[1] else image.shape[1]
    #     w = w + 2*offset if (w + 2*offset) < image.shape[0] else image.shape[0]

    #     face = image[x:x+w, y:y+h]
    #     face_rect_resized = cv2.resize(face, (64, 64))
        
    #     face_rect_normalized = face_rect_resized * 1./255
    #     face_rect_reshape = np.reshape(face_rect_resized, newshape=(1, 64, 64, 3))
    #     y_pred = model.predict(face_rect_reshape)

    #     y_pred = np.array(y_pred)
    #     folder_path = os.path.split(image_path)[0]

    #     print(folder_path)
    #     file_name = image_path.split('/')[-1] 

    #     if y_pred[0, 0] < 0.5:         
    #         new_name = '00_{}'.format(file_name)
    #         new_path = os.path.join(folder_path, new_name)         
    #     else: 
    #         new_name = '01_{}'.format(file_name)
    #         new_path = os.path.join(folder_path, new_name)

    #     # copy(image_path, new_folder)

    #     # new_old_path = os.path.join(new_folder, file_name)  
    #     rename(image_path, new_path)
    #     print(new_path)
    #     break

    file_name = image_path.split('/')[-1] 

    # # if file_name.split('_')[1] = '00':

    # #     new_name = '01_{}'.format(file_name)
    # #     new_path = os.path.join(folder_path, new_name)

    # #     rename(image_path, new_path)


    # folder_path = os.path.split(image_path)[0]
    # age = file_name.split('_')[0]

    # age_path = os.path.join(folder_path, age)

    # if not os.path.isdir(age_path):
    #     os.mkdir(age_path)
    # folder1_path = os.path.join(image_path, image_path)
    # 

# Undo remove    
    # for file1 in os.listdir(image_path):
    #     # print(file1)
    #     # print('---------------------------------------------')
    #     file1_path = os.path.join(image_path, file1)
    # move(image_path, output_folder)


# def _move_()
#     folder_path = os.path.split(image_path)[0]
#     age = file_name.split('_')[0]

#     age_path = os.path.join(folder_path, age)

#     if not os.path.isdir(age_path):
#         os.mkdir(age_path)

#     move(image_path, age_path)


# Move file
    folder_path = os.path.split(image_path)[0]
    age = file_name.split('-')[0][1::]

    age_path = os.path.join(folder_path, age)

    if not os.path.isdir(age_path):
        os.mkdir(age_path)

    move(image_path, age_path)
