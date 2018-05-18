import time
import dlib

class Person():
    def __init__(self, person_id, gender=None, age=None):
        self.person_id = person_id
        self.face_tracker = dlib.correlation_tracker()
        self.bbox = None

        self.face_info = 'Detecting...'
        self.cropped_face_arr = []
        self.num_face_in_arr = 0
        self.gender = None
        self.age = None
        self.watching_time = 0
        
        self.AGE_RANGE_ARR = ['0-18', '18-25', '25-35', '35-50', '>50']
        self.GENDER_ARR = ['Male', 'Female']

        # self.t = time.time()    # Start the time whenever initialzing


    def getId(self):
        return self.person_id


    def getFaceTracker(self):
        return self.face_tracker


    def getFaceInfo(self):
        if self.gender is None or self.age is None:
            return 'Detecting...'
        else:
            return 'Person: {} #{} #{}'.format(self.person_id, self.gender, self.age)


    def getCroppedFaceArr(self):
        return self.cropped_face_arr


    def getNumFaceInArr(self):
        return self.num_face_in_arr


    def getGender(self):
        return self.gender


    def getAge(self):
        return self.age


    def setId(self, person_id):
        self.person_id = person_id


    def setFaceTracker(self, face_tracker):
        self.face_tracker = face_tracker


    def setFaceInfo(self, face_name):
        self.face_name = face_name


    def addCroppedFaceArr(self, cropped_face_arr):
        self.cropped_face_arr.append(cropped_face_arr)


    def increase_num_face_in_arr(self):
        self.num_face_in_arr += 1


    def setGender(self, gender):
        self.gender = gender


    def setAge(self, age):
        self.age = age


    def startTrack(self, original_image, bbox):
        (x, y, w, h) = bbox
        offset = int(0.07*w)
        # offset = 0
        self.face_tracker.start_track(original_image, dlib.rectangle(x-offset, 
                                                                     y-offset, 
                                                                     x+w+offset, 
                                                                     y+h+offset))


    def getPosition(self):
        self.bbox = self.face_tracker.get_position()
        t_x = int(self.bbox.left())
        t_y = int(self.bbox.top())
        t_w = int(self.bbox.width())
        t_h = int(self.bbox.height())
        offset = int(0.07*t_w)
        t_x = t_x - offset
        t_y = t_y - offset
        t_w = t_w + 2*offset
        t_h = t_h + 2*offset

        return [t_x, t_y, t_w, t_h]


    def updatePosition(self, original_image):
        trackingQuality = self.face_tracker.update(original_image)
        return trackingQuality