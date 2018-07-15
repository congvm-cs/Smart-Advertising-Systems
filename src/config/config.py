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
WEIGHT_PATH = 'D:/MegaSyns/Projects/Smart-Advertising-Systems/src/pretrain_models/train-weights-model-lastest.h5'
# WEIGHT_PATH = 'D:/MegaSyns/Projects/Smart-Advertising-Systems/weights-model-24-5.h5'
MODEL_ARCHITECTURE_JSON = '/mnt/Data/MegaSyns/Projects/Smart-Advertising-Systems/model_archi.json'

''' Face Detection Method
        DLIB: HOG and SVM
            Histogram of the Gradient and Support Vector Machine
        
        HAAR: HAAR METHOD
'''
DETECTION_METHOD = 'DLIB' # DLIB or HAAR
HAAR_MODEL_PATH = 'D:/MegaSyns/Projects/Smart-Advertising-Systems/src/pretrain_models/haarcascade_frontalface_default.xml'
NUM_IMG_STORED = 15

UPLOAD_ADDRESS = 'http://127.0.0.1:5000/upload_data'

DICT_RESULTS = {
    'startTime': 0,
    'endTime': 0,
    'videoIndex': 0,
    '0-12': {
        'Male': {
            'total': 0,
            'watching_time': 0
        },
        'Female': {
            'total': 0,
            'watching_time': 0
        }
    },

    '12-18': {
        'Male': {
            'total': 0,
            'watching_time': 0
        },
        'Female': {
            'total': 0,
            'watching_time': 0
        }
    },

    '18-25': {
        'Male': {
            'total': 0,
            'watching_time': 0
        },
        'Female': {
            'total': 0,
            'watching_time': 0
        }
    },

    '25-35': {
        'Male': {
            'total': 0,
            'watching_time': 0
        },
        'Female': {
            'total': 0,
            'watching_time': 0
        }
    },

    '35-50': {
        'Male': {
            'total': 0,
            'watching_time': 0
        },
        'Female': {
            'total': 0,
            'watching_time': 0
        }
    },

    '>50': {
        'Male': {
            'total': 0,
            'watching_time': 0
        },
        'Female': {
            'total': 0,
            'watching_time': 0
        }
    }
}