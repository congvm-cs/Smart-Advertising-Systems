''' This is configuration for Smart Ads System
'''
# Input layer
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_DEPTH = 3

# Last layer
OUTPUT_GENDER = 1
OUTPUT_AGE = 5

# Pretrain Model
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
WEIGHT_PATH = 'D:\MegaSyns\Projects\Smart-Advertising-Systems\train-weights-model-lastest.h5'
=======
WEIGHT_PATH = '/Users/ngocphu/Smart-Advertising-Systems/train-weights-model-lastest.h5'
>>>>>>> Stashed changes
=======
WEIGHT_PATH = 'D:\\MegaSyns\\Projects\\Smart-Advertising-Systems\\train-weights-model-lastest.h5'
>>>>>>> Stashed changes
=======
WEIGHT_PATH = 'D:\\MegaSyns\\Projects\\Smart-Advertising-Systems\\train-weights-model-lastest.h5'
>>>>>>> Stashed changes
MODEL_ARCHITECTURE_JSON = '/mnt/Data/MegaSyns/Projects/Smart-Advertising-Systems/model_archi.json'

''' Face Detection Method
        DLIB: HOG and SVM
            Histogram of the Gradient and Support Vector Machine
        
        HAAR: HAAR METHOD
'''
DETECTION_METHOD = 'HAAR' # DLIB or HAAR

<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
HAAR_MODEL_PATH = 'D:\MegaSyns\Projects\Smart-Advertising-Systems\Models\haarcascade_frontalface_default.xml'
=======
HAAR_MODEL_PATH = '/Users/ngocphu/Smart-Advertising-Systems/Models/haarcascade_frontalface_default.xml'
>>>>>>> Stashed changes
=======
HAAR_MODEL_PATH = 'D:\\MegaSyns\\Projects\\Smart-Advertising-Systems\\Models\\haarcascade_frontalface_default.xml'
>>>>>>> Stashed changes
=======
HAAR_MODEL_PATH = 'D:\\MegaSyns\\Projects\\Smart-Advertising-Systems\\Models\\haarcascade_frontalface_default.xml'
>>>>>>> Stashed changes

NUM_IMG_STORED = 15