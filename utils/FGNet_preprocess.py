from MTCNN import MTCNN
import argparse
import os
import cv2
from shutil import move


def split_train_test(data_dir, test):

    pass

def main(args):
    input_data_dir = args.input_data_dir
    output_data_dir = args.output_data_dir
    image_size = args.image_size
    mtcnn_model_dir = args.mtcnn_model_dir

    mtcnn = MTCNN.MTCNN(mtcnn_model_dir)
    
    print('Loading image from: {}'.format(input_data_dir))

    for folder_name in os.listdir(input_data_dir):
        folder_path = os.path.join(input_data_dir, folder_name)
        # output_folder_path = os.path.join(output_data_dir, folder_name)

        # if not os.path.isdir(output_folder_path):
        #     os.mkdir(output_folder_path)

        print('>>> Loading image from folder: {}'.format(folder_name))
        print('New file')

        for file_name in os.listdir(folder_path):
            # print(str(file_name).split('.')[1])
            
            # if str(file_name)[-3:] == 'JPG':
            file_path = os.path.join(folder_path, file_name)

            print('#File: {}'.format(file_path))

            img_origin = cv2.imread(str(file_path))
            face_crop, ret = mtcnn.single_face_crop(img_origin)
        
            if ret == False:
                continue
            else:            
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                resized_face = cv2.resize(face_crop, (image_size, image_size))
                cv2.imwrite(os.path.join(output_data_dir, file_name), resized_face)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mtcnn_model_dir', help='mtcnn model directory', default=None, type=str)
    parser.add_argument('--input_data_dir', help='input data directory', default=None, type=str)
    parser.add_argument('--output_data_dir', help='output data directory', default=None, type=str)
    parser.add_argument('--image_size', help='image size', default=64, type=int)
    args = parser.parse_args()
    main(args)
