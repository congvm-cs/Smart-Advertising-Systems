#import the OpenCV and dlib libraries
import sys
sys.path.append('.')
sys.path.append('./Models/')

import cv2
import dlib
import threading
import time
import numpy as np
import agutils
import agconfig
import FaceDetetion
import AGNet
import Person

class MultiTracking():

    def __init__(self):

        self.agModel = AGNet.AGNet(verbose=False)
        self.detector = FaceDetetion.FaceDetection()    
    
        self.OUTPUT_SIZE_WIDTH = 775
        self.OUTPUT_SIZE_HEIGHT = 600
        self.rectangleColor = (0, 255, 0)
        self.fps = 0

        # Fixed Soon
        self.currentFaceID = 0
        self.faceTrackers = {}
        self.faceNames = {}
        self.faceArr = {}
        self.numEveryFaceInDict = {}
        self.baseImage = None


        # New Class
        self.PersonManager = []


    def doRecognizePerson(self, person):
        print('Start predict')
        # Predict gender and age here
        # collect 10 faces to predict exactly
        [gender_pred, age_pred] = self.agModel.predict_with_array(person.getCroppedFaceArr())
        person.setGender(gender_pred)
        person.setAge(age_pred)



    def check_new_face(self):
        #For the face detection, we need to make use of a gray
        #colored image so we will convert the baseImage to a
        #gray-based image
        gray = cv2.cvtColor(self.baseImage, cv2.COLOR_BGR2GRAY)
        
        #Now use the FaceDetection detector to find all faces
        faces = self.detector.detectMultiFaces(gray)

        for bbox in faces:
            (x, y, w, h) = bbox

            #calculate the centerpoint
            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h

            #Variable holding information which faceid we matched with
            matchedFid = False

            #Now loop over all the trackers and check if the 
            #centerpoint of the face is within the box of a 
            #tracker
            for person in self.PersonManager:
                [t_x, t_y, t_w, t_h] = person.getPosition()

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
                    matchedFid = True
                    # Keep prediction on fid


#===============================================CREATE NEW FACE========================================#

            if matchedFid is False:
                print("Creating new tracker " + str(self.currentFaceID))

                #---------------------------------------------------------------------------------------#
                person = Person.Person(self.currentFaceID)
                person.startTrack(self.baseImage, bbox)

                self.PersonManager.append(person)
                #---------------------------------------------------------------------------------------#

                #Increase the currentFaceID counter
                self.currentFaceID += 1


#=====================================================================================================#
    def detectAndTrackMultipleFaces(self):
        #Open the first webcame device
        capture = cv2.VideoCapture(1)

        #Create two opencv named windows
        cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

        #Position the windows next to eachother
        cv2.moveWindow("base-image", 0, 100)
        cv2.moveWindow("result-image", 400, 100)

        #variables holding the current frame number and the current faceid
        frameCounter = 0

#=====================================================================================================#*
        try:
            while True:
                # Start timer
                timer = cv2.getTickCount()

                #Retrieve the latest image from the webcam
                ret, fullSizeBaseImage = capture.read()

                #Resize the image to 320x240
                self.baseImage = cv2.resize(fullSizeBaseImage, (0, 0), fx=0.5, fy=0.5)
                resultImage = self.baseImage.copy()

                # self.baseImage = cv2.cvtColor(self.baseImage, cv2.COLOR_BGR2RGB)

                #Check if a key was pressed and if it was Q, then break
                #from the infinite loop
                pressedKey = cv2.waitKey(1)
                if pressedKey == ord('q'):
                    break

                #Result image is the image we will show the user, which is a
                #combination of the original image from the webcam and the
                #overlayed rectangle for the largest face
                

                #Increase the framecounter
                frameCounter += 1 

                #Update all the trackers and remove the ones for which the update
                #indicated the quality was not good enough
                fidsToDelete = []

#=====================================================================================================#*            
                for person in self.PersonManager:
                    [t_x, t_y, t_w, t_h] = person.getPosition()

                    t_x = agutils.saturation(t_x, 0, self.baseImage.shape[1])
                    t_y = agutils.saturation(t_y, 0, self.baseImage.shape[0])
                    t_w = int(t_w)
                    t_h = int(t_h)

                    agutils.draw_rectangle(resultImage, t_x, t_y, t_x + t_w, t_y + t_h, self.rectangleColor)

                    cv2.putText(resultImage, person.getFaceInfo() , 
                                (int(t_x + t_w/2), int(t_y)), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 2)

#=====================================================================================================#*
                    #If the tracking quality is not good enough, we must delete
                    #this tracker
                    trackingQuality = person.updatePosition(self.baseImage)

                    if trackingQuality < 6:
                        fidsToDelete.append(person)
                        
                    
                    if person.getNumFaceInArr() < 15:
                        crop_image = self.baseImage[t_y:t_y+t_h, 
                                                    t_x:t_x+t_w, :]
                        
                        
                        crop_image_resized = cv2.resize(crop_image, (agconfig.IMAGE_WIDTH, agconfig.IMAGE_HEIGHT))

                        # if t_w > self.baseImage.shape[0]/3:
                        #     crop_image_resized = cv2.blur(crop_image_resized, (3, 3))
                        
                        cv2.imshow('hi', crop_image_resized)
                        person.addCroppedFaceArr(crop_image_resized)
                        person.increase_num_face_in_arr()


                    elif person.getNumFaceInArr() == 15:
                        t = threading.Thread(target = self.doRecognizePerson,
                                             args=([person]))

                        t.start()
                        t.join()

                        person.increase_num_face_in_arr()


#=====================================================================================================#*    
                for person in fidsToDelete:
                    print("Removing fid " + str(person.getId()) + " from list of trackers")
                    # self.faceTrackers.pop(fid , None )
                    # self.numEveryFaceInDict.pop(fid, None)
                    # self.faceArr.pop(fid, None)
                    self.PersonManager.remove(person)

#=====================================================================================================#*

                #Every 10 frames, we will have to determine which faces
                #are present in the frame
                if (frameCounter % 10) == 0:
                    t2 = threading.Thread(target=self.check_new_face)
                    t2.start()

                    # Calculate Frames per second (FPS)
                    self.fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
                
                # Display FPS on frame
                cv2.putText(resultImage, "FPS : " + str(int(self.fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                #Since we want to show something larger on the screen than the
                #original 320x240, we resize the image again
                #
                #Note that it would also be possible to keep the large version
                #of the baseimage and make the result image a copy of this large
                #base image and use the scaling factor to draw the rectangle
                #at the right coordinates.


#================================================Visualizing=====================================================#*
                largeResult = cv2.resize(resultImage,
                                        (self.OUTPUT_SIZE_WIDTH, self.OUTPUT_SIZE_HEIGHT))

                #Finally, we want to show the images on the screen
                cv2.imshow("base-image", self.baseImage)
                cv2.imshow("result-image", largeResult)



        #To ensure we can also deal with the user pressing Ctrl-C in the console
        #we have to check for the KeyboardInterrupt exception and break out of
        #the main loop
        except KeyboardInterrupt as e:
            pass

        #Destroy any OpenCV windows and exit the application
        cv2.destroyAllWindows()


if __name__ == '__main__':
    tracking = MultiTracking()
    tracking.detectAndTrackMultipleFaces()
