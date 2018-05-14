#import the OpenCV and dlib libraries
import sys
sys.path.append('.')
sys.path.append('./Models/')
import cv2
import dlib
import tensorflow as tf
import threading
import time
import numpy as np


import keras.utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K

import utils
import Config
import FaceDetetion

#Initialize a face cascade using the frontal face haar cascade provided with
#the OpenCV library
#Make sure that you copy this file from the opencv project to the root of this
#project folder


detector = FaceDetetion.FaceDetection()
    
#The deisred output width and height
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600
WEIGHT_PATH = '/mnt/Data/MegaSyns/Projects/Smart-Advertising-Systems/train-weights-model-lastest.h5'

GENDER = ['Male', 'Female']
AGE = ['0-18', '18-25', '25-35', '35-60', '>60']

graph = None

currentFaceID = 0

#Variables holding the correlation trackers and the name per faceid
faceTrackers = {}
faceNames = {}
faceArr = {}
numEveryFaceInDict = {}
baseImage = None


def model_predict(images, model):
    gender_sum = 0
    age_sum = [0, 0, 0, 0, 0]

    for img in images:
        face_rect_resized = cv2.resize(img, (64, 64))

        face_rect_reshape = np.reshape(face_rect_resized, newshape=(1, 64, 64, 3))
        face_rect_reshape = utils.preprocess_image(face_rect_reshape)

        global graph
        with graph.as_default():
            [y_gender_pred, y_age_pred] = model.predict(face_rect_reshape)
        
        gender_sum += y_gender_pred[-1]
        age_sum += y_age_pred

    gender_pred = GENDER[int(np.round(gender_sum/15))]
    age_pred = AGE[np.argmax(y_age_pred)]

    return [gender_pred, age_pred]


#We are not doing really face recognition
def doRecognizePerson(faceNames, fid, images, model):
    # time.sleep(2
    print('Start predict 1')
    # Predict gender and age here

    # collect 10 faces to predict exactly

    # face_arr.append()
    # if len(face_arr) == 10:
    # while True:
    #     pass
    [gender_pred, age_pred] = model_predict(images, model)
        
    faceNames[fid] = "Person {}: {} {}".format(str(fid), gender_pred, age_pred)


def contruct_model():
    global graph
    input_x = Input((64, 64, 3))
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)

    x = Dense(128, activation='relu')(x)

    output_gender = Dense(1, activation='sigmoid', name='gender_output')(x)
    output_age = Dense(5, activation='softmax', name='age_output')(x)

    model = Model(input_x, [output_gender, output_age])
    model.load_weights(WEIGHT_PATH)
    print(model.summary())

    graph = tf.get_default_graph()

    return model


def check_new_face():
    global baseImage, faceTrackers, faceArr, numEveryFaceInDict, currentFaceID

    #For the face detection, we need to make use of a gray
    #colored image so we will convert the baseImage to a
    #gray-based image
    gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
    
    #Now use the FaceDetection detector to find all faces
    #in the image
    faces = detector.detectMultiFaces(gray)

    #Loop over all faces and check if the area for this
    #face is the largest so far
    #We need to convert it to int here because of the
    #requirement of the dlib tracker. If we omit the cast to
    #int here, you will get cast errors since the detector
    #returns numpy.int32 and the tracker requires an int
    for face in faces:
        # if Config.DETECTION_METHOD == 'DLIB':
        #     (x, y, w, h) = rect_to_bb(face) 
        # elif Config.DETECTION_METHOD == 'HAAR':
        
        (x, y, w, h) = face

        #calculate the centerpoint
        x_bar = x + 0.5 * w
        y_bar = y + 0.5 * h

        #Variable holding information which faceid we 
        #matched with
        matchedFid = None

        #Now loop over all the trackers and check if the 
        #centerpoint of the face is within the box of a 
        #tracker
        for fid in faceTrackers.keys():
            tracked_position = faceTrackers[fid].get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())

            #calculate the centerpoint
            t_x_bar = t_x + 0.5 * t_w
            t_y_bar = t_y + 0.5 * t_h

            #check if the centerpoint of the face is within the 
            #rectangleof a tracker region. Also, the centerpoint
            #of the tracker region must be within the region 
            #detected as a face. If both of these conditions hold
            #we have a match
            if (( t_x <= x_bar   <= (t_x + t_w)) and 
                ( t_y <= y_bar   <= (t_y + t_h)) and 
                ( x   <= t_x_bar <= (x   + w  )) and 
                ( y   <= t_y_bar <= (y   + h  ))):
                matchedFid = fid
                # Keep prediction on fid


        #If no matched fid, then we have to create a new tracker
        if matchedFid is None:
            print("Creating new tracker " + str(currentFaceID))
            #Create and store the tracker 
            tracker = dlib.correlation_tracker()
            tracker.start_track(baseImage,
                                dlib.rectangle( x-10,
                                                y-20,
                                                x+w+10,
                                                y+h+20))

            faceTrackers[ currentFaceID ] = tracker

            faceArr[currentFaceID] = []
            numEveryFaceInDict[currentFaceID] = 0
            
            #Increase the currentFaceID counter
            currentFaceID += 1


def detectAndTrackMultipleFaces():
    global baseImage, faceTrackers, faceArr, numEveryFaceInDict, currentFaceID
    # load model
    model = contruct_model()

    #Open the first webcame device
    capture = cv2.VideoCapture(0)

    #Create two opencv named windows
    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    #Position the windows next to eachother
    cv2.moveWindow("base-image", 0, 100)
    cv2.moveWindow("result-image", 400, 100)

    #Start the window thread for the two windows we are using
    cv2.startWindowThread()

    #The color of the rectangle we draw around the face
    rectangleColor = (0, 255, 0)

    #variables holding the current frame number and the current faceid
    frameCounter = 0


    try:
        while True:

            # Start timer
            timer = cv2.getTickCount()

            #Retrieve the latest image from the webcam
            rc, fullSizeBaseImage = capture.read()

            #Resize the image to 320x240
            baseImage = cv2.resize( fullSizeBaseImage, ( 0, 0), fx=0.5, fy=0.5)

            #Check if a key was pressed and if it was Q, then break
            #from the infinite loop
            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('q'):
                break

            #Result image is the image we will show the user, which is a
            #combination of the original image from the webcam and the
            #overlayed rectangle for the largest face
            resultImage = baseImage.copy()

            #STEPS:
            # * Update all trackers and remove the ones that are not 
            #   relevant anymore
            # * Every 10 frames:
            #       + Use face detection on the current frame and look
            #         for faces. 
            #       + For each found face, check if centerpoint is within
            #         existing tracked box. If so, nothing to do
            #       + If centerpoint is NOT in existing tracked box, then
            #         we add a new tracker with a new face-id


            #Increase the framecounter
            frameCounter += 1 

            #Update all the trackers and remove the ones for which the update
            #indicated the quality was not good enough
            fidsToDelete = []
            for fid in faceTrackers.keys():

                #Now loop over all the trackers we have and draw the rectangle
                #around the detected faces. If we 'know' the name for this person
                #(i.e. the recognition thread is finished), we print the name
                #of the person, otherwise the message indicating we are detecting
                #the name of the person
                tracked_position = faceTrackers[fid].get_position()

                t_x = utils.saturation(int(tracked_position.left()), 0, baseImage.shape[1])
                t_y = utils.saturation(int(tracked_position.top()), 0, baseImage.shape[0])
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                utils.draw_rectangle(resultImage, t_x, t_y, t_x + t_w, t_y + t_h, rectangleColor)

                if fid in faceNames.keys():
                    cv2.putText(resultImage, faceNames[fid] , 
                                (int(t_x + t_w/2), int(t_y)), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 2)
                else:
                    cv2.putText(resultImage, "Detecting..." , 
                                (int(t_x + t_w/2), int(t_y)), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 2)


                #If the tracking quality is not good enough, we must delete
                #this tracker
                trackingQuality = faceTrackers[ fid ].update(baseImage)

                if trackingQuality < 6:
                    fidsToDelete.append( fid )
                
                if numEveryFaceInDict[fid] < 15:

                    crop_image = baseImage[t_y:t_y+t_h, t_x:t_x+t_w, :]
                    faceArr[fid].extend([crop_image])
                    numEveryFaceInDict[fid] += 1

                elif numEveryFaceInDict[fid] == 15:
                    t = threading.Thread(  target = doRecognizePerson ,
                                            args=(faceNames, fid, faceArr[fid], model))

                    t.start()
                    numEveryFaceInDict[fid] = 16  # Stop predict


            for fid in fidsToDelete:
                print("Removing fid " + str(fid) + " from list of trackers")
                faceTrackers.pop(fid , None )
                numEveryFaceInDict.pop(fid, None)
                faceArr.pop(fid, None)

            #Every 10 frames, we will have to determine which faces
            #are present in the frame
            if (frameCounter % 10) == 0:
                t2 = threading.Thread(target=check_new_face)
                t2.start()

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
             # Display FPS on frame
            cv2.putText(resultImage, "FPS : " + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            #Since we want to show something larger on the screen than the
            #original 320x240, we resize the image again
            #
            #Note that it would also be possible to keep the large version
            #of the baseimage and make the result image a copy of this large
            #base image and use the scaling factor to draw the rectangle
            #at the right coordinates.

            largeResult = cv2.resize(resultImage,
                                     (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

            #Finally, we want to show the images on the screen
            cv2.imshow("base-image", baseImage)
            cv2.imshow("result-image", largeResult)


            # Calculate Frames per second (FPS)
            fps += cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    #To ensure we can also deal with the user pressing Ctrl-C in the console
    #we have to check for the KeyboardInterrupt exception and break out of
    #the main loop
    except KeyboardInterrupt as e:
        pass

    #Destroy any OpenCV windows and exit the application
    cv2.destroyAllWindows()


# if __name__ == '__main__':
#     detectAndTrackMultipleFaces()
