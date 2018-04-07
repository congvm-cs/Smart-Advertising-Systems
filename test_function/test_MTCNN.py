import os
import sys
sys.path.append(os.path.abspath('..'))
import cv2
from MTCNN import MTCNN

mtcnn = MTCNN.MTCNN('/mnt/Data/MegaSyns/Projects/Smart-Advertising-Systems/MTCNN/Models')

# img = cv2.imread('/mnt/Data/Dataset/Face_Data/The_Images_of_Groups_Dataset/GA_Net_database/A1-G1-2-901573590_e232f54f41_1190_8584264@N02-Wed3a.jpg')
img = cv2.imread('/mnt/Data/Dataset/Face_Data/Output/A1-G2-0-1161504315_fd5eba060a_1431_61275805@N00-Fam2a.jpg')
ret, bboxes, points = mtcnn.__landmark_detect__(img)

print(len(points))
for bb in bboxes:
    bb = bb.astype(int)
    cv2.rectangle(img, (bb[0],bb[1]),(bb[2],bb[3]),(0,255,0),2)

for i in range(int(len(points)/2)):
    cv2.circle(img, (int(points[2*i]), int(int(points[2*i+1]))), 2, (0,0,255), 2)
    
cv2.imshow("hello", img)
cv2.waitKey(0)
