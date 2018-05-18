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

class MultiTracking():

    def __init__(self):
        
        self.OUTPUT_SIZE_WIDTH = 775
        self.OUTPUT_SIZE_HEIGHT = 600

        self.currentFaceID = 0
        self.faceTrackers = {}
        self.faceNames = {}
        self.faceArr = {}
        self.numEveryFaceInDict = {}
        self.baseImage = None

        self.agModel = AGNet.AGNet(verbose=False)
        self.detector = FaceDetetion.FaceDetection()    

        #The color of the rectangle we draw around the face
        self.rectangleColor = (0, 255, 0)

        video_index = 1
        views_on_video = 0
        watching_time = 0


    def doRecognizePerson(self, faceNames, fid, images):
        print('Start predict')
        # Predict gender and age here
        # collect 10 faces to predict exactly
        [gender_pred, age_pred] = self.agModel.predict_with_array(images)

        self.faceNames[fid] = "Person {}: {} {}".format(str(fid), gender_pred, age_pred)


    def check_new_face(self):
        #For the face detection, we need to make use of a gray
        #colored image so we will convert the baseImage to a
        #gray-based image
        gray = cv2.cvtColor(self.baseImage, cv2.COLOR_BGR2GRAY)
        
        #Now use the FaceDetection detector to find all faces
        faces = self.detector.detectMultiFaces(gray)

        for face in faces:
            (x, y, w, h) = face

            #calculate the centerpoint
            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h

            #Variable holding information which faceid we matched with
            matchedFid = None

            #Now loop over all the trackers and check if the 
            #centerpoint of the face is within the box of a 
            #tracker
            for fid in self.faceTrackers.keys():
                tracked_position = self.faceTrackers[fid].get_position()

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
                print("Creating new tracker " + str(self.currentFaceID))

                #Create and store the tracker 
                tracker = dlib.correlation_tracker()

                offset = int(0.15*w)
                tracker.start_track(self.baseImage,
                                    dlib.rectangle( x-offset, 
                                                    y-offset, 
                                                    x+w+offset, 
                                                    y+h+offset))

                self.faceTrackers[self.currentFaceID] = tracker

                self.faceArr[self.currentFaceID] = []
                self.numEveryFaceInDict[self.currentFaceID] = 0
                
                #Increase the currentFaceID counter
                self.currentFaceID += 1


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

        try:
            while True:
                # Start timer
                timer = cv2.getTickCount()

                #Retrieve the latest image from the webcam
                ret, fullSizeBaseImage = capture.read()

                #Resize the image to 320x240
                self.baseImage = cv2.resize(fullSizeBaseImage, (0, 0), fx=0.5, fy=0.5)

                #Check if a key was pressed and if it was Q, then break
                #from the infinite loop
                pressedKey = cv2.waitKey(1)
                if pressedKey == ord('q'):
                    break

                #Result image is the image we will show the user, which is a
                #combination of the original image from the webcam and the
                #overlayed rectangle for the largest face
                resultImage = self.baseImage.copy()

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
                
                for fid in self.faceTrackers.keys():

                    #Now loop over all the trackers we have and draw the rectangle
                    #around the detected faces. If we 'know' the name for this person
                    #(i.e. the recognition thread is finished), we print the name
                    #of the person, otherwise the message indicating we are detecting
                    #the name of the person
                    tracked_position = self.faceTrackers[fid].get_position()

                    t_x = agutils.saturation(int(tracked_position.left()), 0, self.baseImage.shape[1])
                    t_y = agutils.saturation(int(tracked_position.top()), 0, self.baseImage.shape[0])
                    t_w = int(tracked_position.width())
                    t_h = int(tracked_position.height())

                    agutils.draw_rectangle(resultImage, t_x, t_y, t_x + t_w, t_y + t_h, self.rectangleColor)

                    if fid in self.faceNames.keys():
                        cv2.putText(resultImage, self.faceNames[fid] , 
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
                    trackingQuality = self.faceTrackers[fid].update(self.baseImage)

                    if trackingQuality < 6:
                        fidsToDelete.append(fid)
                    
                    if self.numEveryFaceInDict[fid] < 15:
                        crop_image = self.baseImage[t_y:t_y+t_h, 
                                                    t_x:t_x+t_w, :]

                        self.faceArr[fid].extend([crop_image])
                        self.numEveryFaceInDict[fid] += 1

                    elif self.numEveryFaceInDict[fid] == 15:
                        t = threading.Thread(target = self.doRecognizePerson,
                                             args=(self.faceNames, fid, self.faceArr[fid]))

                        t.start()

                        self.numEveryFaceInDict[fid] = 16  # Stop predict


                
                for fid in fidsToDelete:
                    print("Removing fid " + str(fid) + " from list of trackers")
                    self.faceTrackers.pop(fid , None )
                    self.numEveryFaceInDict.pop(fid, None)
                    self.faceArr.pop(fid, None)

                

                #Every 10 frames, we will have to determine which faces
                #are present in the frame
                if (frameCounter % 10) == 0:
                    t2 = threading.Thread(target=self.check_new_face)
                    t2.start()
                    # t2.join()

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
                
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
                                        (self.OUTPUT_SIZE_WIDTH, self.OUTPUT_SIZE_HEIGHT))

                #Finally, we want to show the images on the screen
                cv2.imshow("base-image", self.baseImage)
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


if __name__ == '__main__':
    tracking = MultiTracking()
    tracking.detectAndTrackMultipleFaces()
