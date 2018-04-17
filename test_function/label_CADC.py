import os
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
model_path = '/Users/ngocphu/Documents/FINAL_PROJECT_RESEARCH/Smart_Advertising_Systems/AGNet_weights_1-improvement-17-0.20-0.92.hdf5'
align_predictor_path = '/Users/ngocphu/Documents/FINAL_PROJECT_RESEARCH/shape_predictor_68_face_landmarks.dat'
input_data_dir = '/Users/ngocphu/Documents/FINAL_PROJECT_RESEARCH/CACD2000'
file_test = '/Users/ngocphu/Documents/FINAL_PROJECT_RESEARCH/CACD2000/14_Alex_Pettyfer_0007.jpg'
output_path = '/Users/ngocphu/Documents/FINAL_PROJECT_RESEARCH/label_CACD2000'
# Camera Streaming
# cap = cv2.VideoCapture(0)
# cap = WebcamVideoStream(src=0).start()
# fps = FPS().start()

detector = dlib.get_frontal_face_detector()
model = load_model(model_path)
predictor = dlib.shape_predictor(align_predictor_path)
fa = FaceAligner(predictor, desiredFaceWidth=64)
NUM_FRAMES = 100


# ret, frame = cap.read()
#frame = cap.read()
# image = frame.copy()
#image = cv2.imread(file_test)
#cv2.imshow('aaa',image)
#image = np.asarray(image)

            
for file_name in os.listdir(input_data_dir):
    split_name = str(file_name).split('_')
    file_path = os.path.join(input_data_dir, file_name)
    
    print('#File: {}'.format(file_path))
    image_origin = cv2.imread(str(file_path))
    tmp_img = image_origin.copy()
    gray = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    for (i, rect) in enumerate(rects):
        (x, y, h, w) = rect_to_bb(rect)             # Positions of rectangle contains face
        print (x,y,h,w)
        offset = int(0.1*x) 
        x = x - offset
        if x < 0:
            x = 0
        y = y - offset
        if y < 0: 
            y = 0
        h = h + 2*offset
        w = w + 2*offset

        face = image_origin[x:x+w, y:y+h]
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
            gender = '_0_'
        else: 
            gender = '_1_'

        new_name = split_name[0] + gender + file_name
        cv2.imwrite(os.path.join(output_path, new_name), image_origin)
        
    #     cv2.putText(tmp_img, "#{} #{} #{:2f}".format(i+1, gender, float(y_pred[0, 0])), (x-10, y-20),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)       
            
    #     cv2.rectangle(tmp_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # cv2.imshow('detect', tmp_img)
    # key = cv2.waitKey(0)
    # if key == ord('a'):
    #     cv2.destroyAllWindows() 
    #     continue

#cap.stop()