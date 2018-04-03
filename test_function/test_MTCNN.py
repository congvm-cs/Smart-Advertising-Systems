import os
import sys
sys.path.append(os.path.abspath(..))
import cv2
from MTCNN import MTCNN

mtcnn = MTCNN.MTCNN('/mnt/Data/MegaSyns/Projects/Smart-Advertising-Systems/MTCNN/Models')

img = cv2.imread()
ret, bboxes, points = mtcnn.__landmark_detect__(img)

