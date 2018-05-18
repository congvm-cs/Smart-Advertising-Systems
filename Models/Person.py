class Person():
    def __init__(self):
        self.person_id = 0
        self.face_tracker = None
        self.face_name = None
        self.cropped_face_arr = None
        self.num_face_in_arr = 0


    def getId(self):
        return self.person_id


    def getFaceTracker(self):
        return self.face_tracker


    def getFaceName(self):
        return self.face_name

    def getCroppedFaceArr(self):
        return self.cropped_face_arr

    def getNumFaceInArr(self):
        return self.num_face_in_arr


    def setId(self, person_id):
        self.person_id = person_id


    def setFaceTracker(self, face_tracker):
        self.face_tracker = face_tracker


    def setFaceName(self, face_name):
        self.face_name = face_name


    def setCroppedFaceArr(self, cropped_face_arr):
        self.cropped_face_arr = cropped_face_arr


    def setNumFaceInArr(self, num_face_in_arr):
        self.num_face_in_arr = num_face_in_arr

