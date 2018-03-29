"""Pipeline
    -> Load filename 
    -> Check isfilename? --T--> Load image -> Crop faces
                         --F--> Load coordinations
    -> Check coordinations isinside? --T--> corresponding label --> Save cropped image
                                     --F--> Continue 
"""

import cv2
import os
from MTCNN import MTCNN

file_paths = ['/media/vmc/12D37C49724FE954/Face_Data/The_Images_of_Groups_Dataset/Fam2a/PersonData.txt',
            '/media/vmc/12D37C49724FE954/Face_Data/The_Images_of_Groups_Dataset/Fam4a/PersonData.txt',
            '/media/vmc/12D37C49724FE954/Face_Data/The_Images_of_Groups_Dataset/Fam5a/PersonData.txt',
            '/media/vmc/12D37C49724FE954/Face_Data/The_Images_of_Groups_Dataset/Fam8a/PersonData.txt',
            '/media/vmc/12D37C49724FE954/Face_Data/The_Images_of_Groups_Dataset/Group2a/PersonData.txt',
            '/media/vmc/12D37C49724FE954/Face_Data/The_Images_of_Groups_Dataset/Group4a/PersonData.txt',
            '/media/vmc/12D37C49724FE954/Face_Data/The_Images_of_Groups_Dataset/Group5a/PersonData.txt',
            '/media/vmc/12D37C49724FE954/Face_Data/The_Images_of_Groups_Dataset/Group8a/PersonData.txt',
            '/media/vmc/12D37C49724FE954/Face_Data/The_Images_of_Groups_Dataset/Wed2a/PersonData.txt',
            '/media/vmc/12D37C49724FE954/Face_Data/The_Images_of_Groups_Dataset/Wed3a/PersonData.txt',
            '/media/vmc/12D37C49724FE954/Face_Data/The_Images_of_Groups_Dataset/Wed5a/PersonData.txt'
]

output_path = '/media/vmc/12D37C49724FE954/Face_Data/The_Images_of_Groups_Dataset/Output_face'
img = 0
mtcnn_model_dir = '/media/vmc/12D37C49724FE954/Smart-Advertising-Systems/MTCNN/Models'
mtcnn = MTCNN.MTCNN(mtcnn_model_dir)

def draw(img , coord):
    cv2.circle(img, (coord[0][0], coord[0][1]), 1, (0, 255, 0), 2)
    cv2.circle(img, (coord[1][0], coord[1][1]), 1, (0, 255, 0), 2)
    return img


def IsInside(bounding_boxes, coord):
    pt1 = coord[0]
    pt2 = coord[1]
    try:
        for index, bb in enumerate(bounding_boxes):
            print('bb index: {}'.format(index))
            bb = bb.astype(int)
            if bb[0] < pt1[0] and bb[2] > pt1[0]:
                pass
            else:
                continue

            if bb[1] < pt1[1] and bb[3] > pt1[1]:
                pass
            else:
                continue
            
            if bb[0] < pt2[0] and bb[2] > pt2[0]:
                pass
            else:
                continue
            
            if bb[1] < pt2[1] and bb[3] > pt2[1]:
                pass
            else:
                continue

            return True, index

    except expression as identifier:
        print('Wrong bb')
        return False, -1
    
    return False, -1

def main():
    crop_faces = []
    bounding_boxes = []
    coord = []

    for file_path in file_paths:
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                # print(line)
                # Read Image filename
                if str(line[-4:]) == 'jpg\n':
                    
                    image_name = line[:-5]
                    image_folder = os.path.split(file_paths[0])[0]

                    image_path = os.path.join(image_folder, line[:-1])  # -1 because of filename: *.jpg\n

                    print('#File {}'.format(image_path))
                    try:
                        img = cv2.imread(image_path)
                        # Split face from image
                        [crop_faces, bounding_boxes, ret] = mtcnn.multi_face_crop(img)
                        print('No.faces: {}'.format(len(bounding_boxes)))

                        if len(bounding_boxes) == 0:
                            continue

                    except Exception as e:
                        print('Cannot open file: {}'.format(image_path))

                # Read attribution - x1 | y1 | x2 | y2 | age | gender 
                else:    
                    x1 = int(line.split('\t')[0])
                    y1 = int(line.split('\t')[1])
                    x2 = int(line.split('\t')[2])
                    y2 = int(line.split('\t')[3])
                    age = int(line.split('\t')[4])
                    gender = int(line.split('\t')[5])

                    print('{}-{}-{}-{}'.format(x1, y1, x2, y2))
                    
                    coord = [[x1, y1], [x2, y2]]
                    # # Check inside
                    [ret, index] = IsInside(bounding_boxes, coord)

                    if ret == True:
                        print('Index : {}'.format(index))

                        try:
                            face_crop = crop_faces[index]
                            face_crop = cv2.resize(face_crop, (64, 64))    
                            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
                            # img = draw(img, coord)
                            # cv2.imshow('test', img)
                            # cv2.waitKey(0)
                            filename = 'A{}-G{}-{}.jpg'.format(age, gender, image_name) 
                            cv2.imwrite(os.path.join(output_path, filename), face_crop)

                        except Exception as e:
                            print('Some faces is too close to edge')
                            print('Fail Image: {}'.format(image_path))
                        # # (face_position[0],face_position[1]),(face_position[2],face_position[3])
                       

if __name__ == '__main__':
    main()
