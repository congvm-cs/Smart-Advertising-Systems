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
        ''' This class is a flow to detect and track frontal 
            multi-face, then regconize their gender and age-range
            
        '''
        self.agModel = AGNet.AGNet(verbose=True)
        self.detector = FaceDetetion.FaceDetection()    
    
        self.OUTPUT_SIZE_WIDTH = 775
        self.OUTPUT_SIZE_HEIGHT = 600
        self.rectangleColor = (0, 255, 0)
        self.fps = 0

        # New Class
        self.PersonManager = []
        self.currentFaceID = 0

        self.watched_time_collector = 0
        self.total_watched_time_stored = 0

        self.views_collector = 0
        self.total_views = 0

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


                #Check if a key was pressed and if it was Q, then break
                #from the infinite loop
                pressedKey = cv2.waitKey(1)
                if pressedKey == ord('q'):
                    break

                #Increase the framecounter
                frameCounter += 1 

                #Update all the trackers and remove the ones for which the update
                #indicated the quality was not good enough
                fidsToDelete = []

                self.views_collector = 0
                self.watched_time_collector = 0

#=====================================================================================================#*            
                for person in self.PersonManager:
                    # Update watched time
                    self.watched_time_collector += person.getWatchingTime()
                    self.views_collector += person.getViews()

                    # Update new position rely on tracker
                    [t_x, t_y, t_w, t_h] = person.getPosition()

                    t_x = agutils.saturation(t_x, 0, self.baseImage.shape[1])
                    t_y = agutils.saturation(t_y, 0, self.baseImage.shape[0])
                    t_w = int(t_w)
                    t_h = int(t_h)

                    agutils.draw_rectangle(resultImage, t_x, t_y, t_x + t_w, t_y + t_h, self.rectangleColor)

                    cv2.putText(resultImage, "Person: " + str(person.getId()) , 
                                (int(t_x), int(t_y - t_h/3)), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                t_w/200, (0, 255, 255), 1)

                    cv2.putText(resultImage, "#: " + str(person.getGender()) , 
                                (int(t_x), int(t_y - t_h/5)), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                t_w/200, (0, 255, 255), 1)

                    cv2.putText(resultImage, "#: " + str(person.getAge()) , 
                                (int(t_x), int(t_y - t_h/19)), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                t_w/200, (0, 255, 255), 1)
#=====================================================================================================#*
                    #If the tracking quality is not good enough, we must delete
                    #this tracker
                    trackingQuality = person.updatePosition(self.baseImage)

                    if trackingQuality < 6:
                        fidsToDelete.append(person)
                        
                    
                    if person.getNumFaceInArr() < agconfig.NUM_IMG_STORED:
                        crop_image = self.baseImage[t_y : t_y+t_h, 
                                                    t_x : t_x+t_w, 
                                                    :]
                        
                        
                        crop_image_resized = cv2.resize(crop_image, (agconfig.IMAGE_WIDTH, agconfig.IMAGE_HEIGHT))

                        
                        cv2.imshow('hi', crop_image_resized)
                        person.addCroppedFaceArr(crop_image_resized)
                        person.increase_num_face_in_arr()

                    elif person.getNumFaceInArr() == agconfig.NUM_IMG_STORED:
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
                    self.total_watched_time_stored += person.getWatchingTime()
                    self.total_views += person.getViews()

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
                cv2.putText(resultImage, "FPS : " + str(int(self.fps)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


                # Update Views and Watching Time
                # self.total_watched_time_stored = self.watched_time_collector
                # self.total_views = self.views_collector

                cv2.putText(resultImage, "Watched Time (sec) : {}".format(str(int(self.total_watched_time_stored))), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                cv2.putText(resultImage, "Views : {}".format(str(self.total_views)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
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
