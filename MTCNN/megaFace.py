from MTCNN import MTCNN
import glob
import cv2
import numpy as np
import os
import dlib
from imutils import face_utils, rect_to_bb


# os.mkdir('/content/output_megaage_asian')
# os.mkdir('/content/output_megaage_asian/train')
# os.mkdir('/content/output_megaage_asian/test')


train_base_data_path = '/content/megaage_asian/train'
# path = glob.glob(data_path)

train_name_arr = np.loadtxt('/content/megaage_asian/list/train_name.txt', dtype = 'str')# test_name_arr = np.loadtxt('/content/megaage/list/test_name.txt')
print(len(train_name_arr))

train_age_arr = np.loadtxt('/content/megaage_asian/list/train_age.txt', dtype = 'int')# test_age_arr = np.loadtxt('/content/megaage/list/test_age.txt')
print(len(train_age_arr))


detector = MTCNN('/content/Smart-Advertising-Systems/MTCNN/Models')

count = 0
ad = 0.1

# For train
for name, age in zip(train_name_arr, train_age_arr):

    path = os.path.join(train_base_data_path, name)
    print('>> {}'.format(path))
    img = cv2.imread(path)

    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(img)

    gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)

    # Now use the haar cascade detector to find all faces# in the image

    crop_face = detector.single_face_crop(img_RGB)

    crop_face_resized = cv2.resize(crop_face, (64, 64))
    crop_face_reshape = np.reshape(crop_face_resized, (-1, 64, 64, 3))

    # Predict
    gender, _ = model.predict(crop_face_reshape)

    gender = int(np.round(float(gender[-1])))

    filename = '{}_{}_{}'.format(age, gender, name)

    write_path = '/content/output_megaage_asian/train/' + filename

    cv2.imwrite(write_path, crop_face_resized)

    print(write_path)
    count += 1

    print('Done by {}/{}'.format(count, len(train_age_arr)))