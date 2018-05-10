#import the OpenCV and dlib libraries
import cv2
import dlib
import tensorflow as tf
from imutils.face_utils import rect_to_bb
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


 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

#Initialize a face cascade using the frontal face haar cascade provided with
#the OpenCV library
#Make sure that you copy this file from the opencv project to the root of this
#project folder
# faceCascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()

#The deisred output width and height
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600
WEIGHT_PATH = '/home/vmc/Downloads/train-weights-model-lastest(2).h5'

GENDER = ['Male', 'Female']
AGE = ['0-18', '18-25', '25-35', '35-60', '>60']

TRACKER_STYLES = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
TRACKER_STYLE = TRACKER_STYLES[4]

graph = None


def saturation(val, min_val, max_val):
    if val > max_val:
        val = max_val
    elif val < min_val:
        val = min_val

    return val

def preprocess_image(img):  
    def __prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y 

    img = img/255.0
    img = __prewhiten(img)
    return img


def model_predict(images, model):
    # (x1, y1, w, h) = rect_to_bb(rect)             # Positions of rectangle contains face
    # x2 = x1 + w
    # y2 = y1 + h

    # offset = int(0.15*w) 
    # x1 = (x1 - offset) if (x1 - offset) > 0 else 0 
    # y1 = (y1 - offset) if (y1 - offset) > 0 else 0
    # y2 = (y2 + offset) if (y2 + offset) < image.shape[0] else image.shape[0]
    # x2 = x2 + offset if (x2 + offset) < image.shape[1] else image.shape[1]
    # crop_im = image[y1:y2, x1:x2, :]
    gender_sum = 0
    age_sum = [0, 0, 0, 0, 0]

    for img in images:
        face_rect_resized = cv2.resize(img, (64, 64))

        face_rect_reshape = np.reshape(face_rect_resized, newshape=(1, 64, 64, 3))
        face_rect_reshape = preprocess_image(face_rect_reshape)

        global graph
        with graph.as_default():
            [y_gender_pred, y_age_pred] = model.predict(face_rect_reshape)
        
        # print(y_gender_pred)
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


def draw_rectangle(img, p1, p2, p3, p4, color):
    offset = int((p3 - p1)/4)
    thickness_heavy_line = 3
    thickness_slim_line = 1

    # Left Top (p1, p2)
    cv2.line(img, (p1, p2), (p1, p2 + offset), color, thickness_heavy_line)
    cv2.line(img, (p1, p2), (p1 + offset, p2 ), color, thickness_heavy_line)
    
    # Left Bottom (p1, p4)
    cv2.line(img, (p1, p4), (p1, p4 - offset), color, thickness_heavy_line)
    cv2.line(img, (p1, p4), (p1 + offset, p4 ), color, thickness_heavy_line)

    # Right Top (p3, p2)
    cv2.line(img, (p3, p2), (p3, p2 + offset), color, thickness_heavy_line)
    cv2.line(img, (p3, p2), (p3 - offset, p2), color, thickness_heavy_line)

    # Right Bottom (p3, p4)
    cv2.line(img, (p3, p4), (p3, p4 - offset), color, thickness_heavy_line)
    cv2.line(img, (p3, p4), (p3 - offset, p4 ), color, thickness_heavy_line)
    
    cv2.line(img, (p1, p2), (p1, p4), color, thickness_slim_line)
    cv2.line(img, (p1, p2), (p3, p2), color, thickness_slim_line)
    cv2.line(img, (p3, p4), (p1, p4), color, thickness_slim_line)
    cv2.line(img, (p3, p4), (p3, p2), color, thickness_slim_line)
    return img



def detectAndTrackMultipleFaces():
    # load model
    model = contruct_model()

    #Open the first webcame device
    capture = cv2.VideoCapture(1)

    #Create two opencv named windows
    # cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    #Position the windows next to eachother
    # cv2.moveWindow("base-image", 0, 100)
    cv2.moveWindow("result-image", 400, 100)

    #Start the window thread for the two windows we are using
    cv2.startWindowThread()

    #The color of the rectangle we draw around the face
    rectangleColor = (0, 255, 0)

    #variables holding the current frame number and the current faceid
    frameCounter = 0
    currentFaceID = 0

    #Variables holding the correlation trackers and the name per faceid
    faceTrackers = {}
    faceNames = {}
    faceArr = {}
    numEveryFaceInDict = {}

    #Update all the trackers and remove the ones for which the update
    #indicated the quality was not good enough

    fidsToDelete = []

    fidsToCreate = []
    
    keepList = []

    fps = 0
    fps_print = 0

    try:
        while True:

            # Start timer
            timer = cv2.getTickCount()

            #Retrieve the latest image from the webcam
            rc, fullSizeBaseImage = capture.read()

            #Resize the image to 320x240
            baseImage = cv2.resize( fullSizeBaseImage, ( 320, 240))

            
            #Check if a key was pressed and if it was Q, then break
            #from the infinite loop
            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('q'):
                break

            #Result image is the image we will show the user, which is a
            #combination of the original image from the webcam and the
            #overlayed rectangle for the largest face
            resultImage = baseImage.copy()

            # Display FPS on frame
            cv2.putText(resultImage, "FPS : " + str(int(fps_print)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

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


            # Update faceTrackers

            # for fid in faceTrackers.keys():
            #     matched = False

            #     for keepFid in keepList:
                    
            #         if fid is keepFid:
            #             matched = True
            #             break

            #     if not matched:
            #         fidsToDelete.append(fid)


            # Delete expired person
            for fid in fidsToDelete:
                print("Removing fid " + str(fid) + " from list of trackers")
                faceTrackers.pop(fid , None )
                numEveryFaceInDict.pop(fid, None)
                faceArr.pop(fid, None)
                faceNames.pop(fid, None)
            
            fidsToDelete.clear()

            # Create new person
            for face in fidsToCreate:

                bbox = rect_to_bb(face)
                print("Creating new tracker " + str(currentFaceID))

                if int(minor_ver) < 3:
                    tracker = cv2.Tracker_create(TRACKER_STYLE)
                else:
                    if TRACKER_STYLE == 'BOOSTING':
                        tracker = cv2.TrackerBoosting_create()
                    if TRACKER_STYLE == 'MIL':
                        tracker = cv2.TrackerMIL_create()
                    if TRACKER_STYLE == 'KCF':
                        tracker = cv2.TrackerKCF_create()
                    if TRACKER_STYLE == 'TLD':
                        tracker = cv2.TrackerTLD_create()
                    if TRACKER_STYLE == 'MEDIANFLOW':
                        tracker = cv2.TrackerMedianFlow_create()
                    if TRACKER_STYLE == 'GOTURN':
                        tracker = cv2.TrackerGOTURN_create()

                ok = tracker.init(baseImage, bbox)

                faceTrackers[ currentFaceID ] = tracker

                faceArr[currentFaceID] = []
                numEveryFaceInDict[currentFaceID] = 0
            
                #Increase the currentFaceID counter
                currentFaceID += 1

            fidsToCreate.clear()

            for fid in faceTrackers.keys():
                # trackingQuality = faceTrackers[ fid ].update(baseImage)
                ok, bbox = faceTrackers[fid].update(baseImage)

                #If the tracking quality is not good enough, we must delete
                #this tracker
                if not ok:
                    fidsToDelete.append(fid)

                ok, bbox = faceTrackers[fid].update(baseImage)

                t_x = saturation(int(bbox[0]), 0, baseImage.shape[1])
                t_y = saturation(int(bbox[1]), 0, baseImage.shape[0])
                t_w = int(bbox[2])
                t_h = int(bbox[3])

                draw_rectangle(resultImage, t_x, t_y, t_x + t_w, t_y + t_h, rectangleColor)
 
                if fid in faceNames.keys():
                    cv2.putText(resultImage, faceNames[fid] , 
                                (int(t_x + t_w/2), int(t_y)), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)
                else:
                    cv2.putText(resultImage, "Detecting..." , 
                                (int(t_x + t_w/2), int(t_y)), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)

                if numEveryFaceInDict[fid] < 15:

                    crop_image = baseImage[t_y:t_y+t_h, t_x:t_x+t_w, :]

                    faceArr[fid].extend([crop_image])
                    numEveryFaceInDict[fid] += 1

                elif numEveryFaceInDict[fid] == 15:
                    t = threading.Thread(  target = doRecognizePerson ,
                                            args=(faceNames, fid, faceArr[fid], model))

                    t.start()
                    numEveryFaceInDict[fid] = 16  # Stop predict

            #Every 10 frames, we will have to determine which faces
            #are present in the frame


            if (frameCounter % 20) == 0:
                
                # Update fpt for printing
                fps_print = fps/20
                fps = 0

                #For the face detection, we need to make use of a gray
                #colored imag so we will convert the baseImage to a
                #gray-based image
                gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
                #Now use the haar cascade detector to find all faces
                #in the image
                #faces = faceCascade.detectMultiScale(gray, 1.3, 6)
                faces = detector(gray, 1)


                #Loop over all faces and check if the area for this
                #face is the largest so far
                #We need to convert it to int here because of the
                #requirement of the dlib tracker. If we omit the cast to
                #int here, you will get cast errors since the detector
                #returns numpy.int32 and the tracker requires an int
            

                print('--------------------------------------------')

                for face in faces:
                    (x, y, w, h) = rect_to_bb(face)  

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
                        # tracked_position = faceTrackers[fid].get_position()
                        print('check {}'.format(fid))
                        ok, bbox = faceTrackers[fid].update(baseImage)

                        t_x = int(bbox[0]) 
                        t_y = int(bbox[1])
                        t_w = int(bbox[2])
                        t_h = int(bbox[3])

                        #calculate the centerpoint
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        #check if the centerpoint of the face is within the 
                        #rectangle of a tracker region. Also, the centerpoint
                        #of the tracker region must be within the region 
                        #detected as a face. If both of these conditions hold
                        #we have a match

                        if ( ( t_x  <= x_bar   <= (t_x + t_w) ) and 
                             ( t_y  <= y_bar   <= (t_y + t_h )) and 
                             ( x    <= t_x_bar <= (x + w)) and 
                             ( y    <= t_y_bar <= (y + h))):
                            matchedFid = fid
                            # keepList.append(fid)
                            # print(len(keepList))
                        # Keep prediction on fid
                        # If bound has no face --> remove
                        
                    #If no matched fid, then we have to create a new tracker
                    if matchedFid is None:
                        print('Create new face')
                        fidsToCreate.append(face)


            #Now loop over all the trackers we have and draw the rectangle
            #around the detected faces. If we 'know' the name for this person
            #(i.e. the recognition thread is finished), we print the name
            #of the person, otherwise the message indicating we are detecting
            #the name of the person
            
            #Since we want to show something larger on the screen than the
            #original 320x240, we resize the image again
            #
            #Note that it would also be possible to keep the large version
            #of the baseimage and make the result image a copy of this large
            #base image and use the scaling factor to draw the rectangle
            #at the right coordinates.

            # Calculate Frames per second (FPS)
            fps += cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            largeResult = cv2.resize(resultImage,
                                     (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

            #Finally, we want to show the images on the screen
            # cv2.imshow("base-image", baseImage)
            cv2.imshow("result-image", largeResult)



    #To ensure we can also deal with the user pressing Ctrl-C in the console
    #we have to check for the KeyboardInterrupt exception and break out of
    #the main loop
    except KeyboardInterrupt as e:
        pass

    #Destroy any OpenCV windows and exit the application
    faceTrackers.clear()
    faceNames.clear()
    faceArr.clear()
    numEveryFaceInDict.clear()
    cv2.destroyAllWindows()
    # exit(0)


if __name__ == '__main__':
    detectAndTrackMultipleFaces()
