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
IMAGE_SIZE = 128

file_paths = [     
        '/mnt/Data/Dataset/Face_Data/The_Images_of_Groups_Dataset/Fam2a/PersonData.txt',
        '/mnt/Data/Dataset/Face_Data/The_Images_of_Groups_Dataset/Fam4a/PersonData.txt',
        '/mnt/Data/Dataset/Face_Data/The_Images_of_Groups_Dataset/Fam5a/PersonData.txt',
        '/mnt/Data/Dataset/Face_Data/The_Images_of_Groups_Dataset/Fam8a/PersonData.txt',
        '/mnt/Data/Dataset/Face_Data/The_Images_of_Groups_Dataset/Group2a/PersonData.txt',                       
        '/mnt/Data/Dataset/Face_Data/The_Images_of_Groups_Dataset/Group4a/PersonData.txt',
        '/mnt/Data/Dataset/Face_Data/The_Images_of_Groups_Dataset/Group5a/PersonData.txt',
        '/mnt/Data/Dataset/Face_Data/The_Images_of_Groups_Dataset/Group8a/PersonData.txt',
        '/mnt/Data/Dataset/Face_Data/The_Images_of_Groups_Dataset/Wed2a/PersonData.txt',
        '/mnt/Data/Dataset/Face_Data/The_Images_of_Groups_Dataset/Wed3a/PersonData.txt',
        '/mnt/Data/Dataset/Face_Data/The_Images_of_Groups_Dataset/Wed5a/PersonData.txt'
]

output_path = '/mnt/Data/Dataset/Face_Data/Output'
img = 0

mtcnn_model_dir = '/mnt/Data/MegaSyns/Projects/Smart-Advertising-Systems/MTCNN/Models'
mtcnn = MTCNN.MTCNN(mtcnn_model_dir)


def draw(img , coord):
    cv2.circle(img, (coord[0][0], coord[0][1]), 1, (0, 255, 0), 2)
    cv2.circle(img, (coord[1][0], coord[1][1]), 1, (0, 255, 0), 2)
    return img


def DrawBB(img, bounding_boxes):
    for bb in bounding_boxes:
        cv2.rectangle(img, (bb[0],bb[1]),(bb[2],bb[3]), (0,255,0),2)
    return img


def IsInside(bounding_boxes, coord):
    pt1 = coord[0]
    pt2 = coord[1]
    
    try:
        for index, bb in enumerate(bounding_boxes):
            # print('bb index: {}'.format(index))
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

    except Exception as identifier:
        print('Wrong bb')
        return False, -1
    
    return False, -1


def main():
    I_crop_faces = []                   # Face image cropped
    bounding_boxes = []                 # Detect facial bounding box
    coord = []                          # Coordination of eye from data's description

    print('#No.Folders: {}'.format(len(file_paths)))

    for file_path in file_paths:

        print('#Open file: {}'.format(file_path))

        file = open(file_path, 'r')
        # with open(file_path, 'r') as file:
        lines = file.readlines()            # Read all lines in file
        
        for line in lines:                   
            # Read Image filename
            if str(line[-4:]) == 'jpg\n':
                I_crop_faces.clear()                 # Reset

                image_name = line[:-5]
                image_folder = os.path.split(file_paths[0])[0]
                image_path = os.path.join(image_folder, line[:-1])  # -1 because of filename: *.jpg\n

                print('--> File {}'.format(image_path))
                try:
                    img = cv2.imread(image_path)
                    # Split face from image
                    [I_crop_faces, bounding_boxes, ret] = mtcnn.multi_face_crop(img)
                    print('No.faces: {}'.format(len(bounding_boxes)))
                    # img = DrawBB(img, bounding_boxes)
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

                # Check whether coord is inside bounding_boxes?
                [ret, index] = IsInside(bounding_boxes, coord)

                if ret == True:
                    print('Index : {}'.format(index))

                    try:
                        face_crop = I_crop_faces[index]
                        face_crop = cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE))    
                        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
                        
                        # img = draw(img, coord)
                        # cv2.imshow('test', img)
                        # cv2.waitKey(0)

                        filename = 'A{}-G{}-{}-{}-{}.jpg'.format(age, gender, index, image_name, image_path.split('/')[6]) 
                        cv2.imwrite(os.path.join(output_path, filename), face_crop)

                    except Exception as e:
                        print('Some faces is too close to edge')
                        print('Fail Image: {}'.format(image_path))
        file.close()
        print('Done')
                       

if __name__ == '__main__':
    main()
