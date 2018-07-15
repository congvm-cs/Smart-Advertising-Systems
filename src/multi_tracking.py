# system modules
import sys
sys.path.append('..')

import cv2
import time
import numpy as np
import threading

# local modules
from src.config import config
from src.ultis import utils
from src import FaceDetetion
from src.models import AGNet
from src import Person
from src.SMLogger import SMLogger


class MultiTracking():
    def __init__(self):
        ''' This class is a flow to detect and track frontal 
            multi-face, then regconize their gender and age-range
        '''
        self.agModel = AGNet.AGNet(verbose=False)
        self.detector = FaceDetetion.FaceDetection()    
    
        self.OUTPUT_SIZE_WIDTH = 640
        self.OUTPUT_SIZE_HEIGHT = 480
        self.rectangleColor = (0, 255, 0)
        self.fps = 0

        # New Class
        self.PersonManager = []
        self.currentFaceID = 0                
        self.fidsToDelete = []
        
        self.watched_time_collector = 0
        self.total_watched_time_stored = 0

        self.views_collector = 0
        self.total_views = 0

        self.baseImage = None
        self.gray = None

        #variables holding the current frame number and the current faceid
        self.frameCounter = 0
        self.video_index = 0

        # Start logging
        self.logger = SMLogger()
        self.logger.open()

        # log content 
        self.dict_results = config.DICT_RESULTS
        self.startTime = time.localtime()


    def doRecognizePerson(self, person):
        print('Start predict')
        # Predict gender and age here
        # collect 10 faces to predict exactly
        [gender_pred, age_pred] = self.agModel.predict_with_array(person.getCroppedFaceArr())
        person.setGender(gender_pred)
        person.setAge(age_pred)
        person.setCollected(True)


    def onVideoChanged(self, video_index):
        return  (self.video_index == video_index)


    def deleteRedundantPerson(self):
        for person in self.fidsToDelete:
            print("Removing fid " + str(person.getId()) + " from list of trackers")
            self.total_watched_time_stored += person.getWatchingTime()
            self.total_views += person.getViews()
            self.PersonManager.remove(person)


    def collectInfo(self):
        for person in self.PersonManager:
            if person.canBeCollected():
                # Collect
                pass

            # else: ignore this person -> move to the other one


    def check_new_face(self):
        #For the face detection, we need to make use of a gray
        #colored image so we will convert the baseImage to a
        #gray-based image
        self.gray = cv2.cvtColor(self.baseImage, cv2.COLOR_BGR2GRAY)
        
        #Now use the FaceDetection detector to find all faces
        faces = self.detector.detectMultiFaces(self.gray)

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

                # check if the centerpoint of the face is within the 
                # rectangle of a tracker region. Also, the centerpoint
                # of the tracker region must be within the region 
                # detected as a face. If both of these conditions hold
                # we have a match
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
    def detectAndTrackMultipleFaces(self, index, frame):

        if self.onVideoChanged(index):
            self.logger.write(self.startTime, self.video_index, self.dict_results)
            self.startTime = time.localtime()

        self.video_index = index
        self.baseImage = frame    
        self.gray = cv2.cvtColor(self.baseImage, cv2.COLOR_BGR2GRAY)

        # while True:
        timer = cv2.getTickCount()
        resultImage = self.baseImage.copy()        
        self.fidsToDelete.clear()

        #Increase the framecounter
        self.frameCounter += 1 
        self.views_collector = 0
        self.watched_time_collector = 0

#=====================================================================================================#*            
        print('[DEBUG][MULTI-TRACKING] UPDATE POSITIONS')
        for person in self.PersonManager:
            # Update watched time
            self.watched_time_collector += person.getWatchingTime()
            self.views_collector += person.getViews()

            # Update new position rely on tracker
            [t_x, t_y, t_w, t_h] = person.getPosition()

            t_x = utils.saturation(t_x, 0, self.baseImage.shape[1])
            t_y = utils.saturation(t_y, 0, self.baseImage.shape[0])
            t_w = int(t_w)
            t_h = int(t_h)

            utils.draw_rectangle(resultImage, t_x, t_y, t_x + t_w, t_y + t_h, self.rectangleColor)

            cv2.putText(resultImage, "Person: " + str(person.getId()) , 
                        (int(t_x), int(t_y - t_h/3)), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        t_w/200, (0, 255, 255), 2)


            cv2.putText(resultImage, "#: " + str(person.getGender()) , 
                        (int(t_x), int(t_y - t_h/5)), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        t_w/200, (0, 255, 255), 2)


            cv2.putText(resultImage, "#: " + str(person.getAge()) , 
                        (int(t_x), int(t_y - t_h/19)), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        t_w/200, (0, 255, 255), 2)


#=====================================================================================================#*
            #If the tracking quality is not good enough, we must delete
            #this tracker
            trackingQuality = person.updatePosition(self.baseImage)
            if trackingQuality < 6:
                self.fidsToDelete.append(person)

            if person.getNumFaceInArr() < config.NUM_IMG_STORED:
                crop_image = self.baseImage[t_y : t_y+t_h, t_x : t_x+t_w, :]
                crop_image_resized = cv2.resize(crop_image, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
                person.addCroppedFaceArr(crop_image_resized)
                person.increase_num_face_in_arr()

            elif person.getNumFaceInArr() == config.NUM_IMG_STORED:
                print('[DEBUG][MULTI] START THREAD DETECT AND TRACKING')
                t = threading.Thread(target = self.doRecognizePerson,
                                        args=([person]))
                t.setDaemon(True)
                t.start()
                t.join()    
                person.increase_num_face_in_arr()
   
        self.deleteRedundantPerson()

        #Every 10 frames, we will have to determine which faces
        #are present in the frame
        if (self.frameCounter % 20) == 0:
            # t2 = threading.Thread(target=self.check_new_face)
            # t2.start()
            print('[DEBUG][MULTI] CHECK NEW FACES')
            #Now use the FaceDetection detector to find all faces
            faces = self.detector.detectMultiFaces(self.gray)

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

                    person = Person.Person(self.currentFaceID)
                    person.startTrack(self.baseImage, bbox)
                    self.PersonManager.append(person)
                    
                    #Increase the currentFaceID counter
                    self.currentFaceID += 1

            # Calculate Frames per second (FPS)
            self.fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
        
        # Display FPS on frame
        cv2.putText(resultImage, "FPS : " + str(int(self.fps)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Display index video
        cv2.putText(resultImage, "Video : " + str(int(self.video_index)), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(resultImage, "Watched Time (sec) : {}".format(str(int(self.total_watched_time_stored))), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(resultImage, "Views : {}".format(str(self.total_views)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

#================================================Visualizing=====================================================#*
        largeResult = cv2.resize(resultImage,
                                (self.OUTPUT_SIZE_WIDTH, self.OUTPUT_SIZE_HEIGHT))

        #Finally, we want to show the images on the screen
        cv2.imshow("result-image", largeResult)
        cv2.waitKey(1)
