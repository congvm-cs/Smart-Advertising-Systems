import dlib
import numpy as np 
from keras.models import load_model
import cv2
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from imutils.face_utils import rect_to_bb, FaceAligner
from keras.preprocessing.image import ImageDataGenerator


print('Load model...')
# model_path = '/mnt/e/Smart-Advertising-Systems-master/Models/weights-improvement-23-0.23-0.92.hdf5'
model_path = '/home/vmc/Desktop/AGNet_weights-improvement-10-0.28-0.87.hdf5'
align_predictor_path = '/mnt/Data/MegaSyns/Projects/Smart-Advertising-Systems/dlib_face_landmarks_model/shape_predictor_68_face_landmarks.dat'
file_dir = '/Users/ngocphu/Documents/FINAL_PROJECT_RESEARCH/CACD2000'
file_test = ''
# Camera Streaming
# cap = cv2.VideoCapture(0)
cap = WebcamVideoStream(src=0).start()
fps = FPS().start()

detector = dlib.get_frontal_face_detector()
model = load_model(model_path)
predictor = dlib.shape_predictor(align_predictor_path)
fa = FaceAligner(predictor, desiredFaceWidth=64)
NUM_FRAMES = 100


# ret, frame = cap.read()
#frame = cap.read()
# image = frame.copy()

image = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)
img_clone = image.copy()

for (i, rect) in enumerate(rects):
    (x, y, h, w) = rect_to_bb(rect)             # Positions of rectangle contains face

    offset = int(0.2*x) 
    x = x - offset
    y = y - offset
    h = h + 2*offset
    w = w + 2*offset

    face = image[x:x+w, y:y+h]
    # faceAligned = fa.align(image, gray, rect)   # Aligned face image 
    # faceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
    face_rect_resized = cv2.resize(face, (64, 64))
    
    # cv2.imshow('after aligned #{}'.format(str(i)), face_rect_resized)
    face_rect_normalized = face_rect_resized * 1./255
    face_rect_reshape = np.reshape(face_rect_normalized, newshape=(1, 64, 64, 3))
    y_pred = model.predict(face_rect_reshape)

    y_pred = np.array(y_pred)
    # print(y_pred.shape)

    # print(type(y_pred))

    print(y_pred)
    if y_pred[0, 0] < 0.5:
        gender = 'Nam'
    else: 
        gender = 'Nu'

    cv2.putText(img_clone, "#{} #{} #{:2f}".format(i+1, gender, float(y_pred[0, 0])), (x-10, y-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)       
    
    
    cv2.rectangle(img_clone, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # fps.update()

    # cv2.putText(img_clone, "FPS: {} ".format(fps.fps()), (10, 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)     

    cv2.imshow('detect', img_clone)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows() 

cap.stop()