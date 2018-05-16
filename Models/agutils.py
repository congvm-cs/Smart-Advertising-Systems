'''This is script for all of tools using in this project
'''
import numpy as np
import cv2
from keras.models import model_from_json
import agconfig 
from PIL import Image
from PIL import ImageTk


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


def draw_rectangle(img, x1, y1, x2, y2, color=(0, 255, 0)):
    ''' Every fancy bounding box instead of bored cv2.rectangle
        Parameter:
            img:    Input image to draw on
            (x1, y1):   Top Left
            (x2, y2):   Bottom Right
            color:  Color of bounding box - default: (0, 255, 0)
    '''

    offset = int((x2 - x1)/4)
    thickness_heavy_line = 3
    thickness_slim_line = 1

    # Draw short line
    # Left Top (x1, y1)
    cv2.line(img, (x1, y1), (x1, y1 + offset), color, thickness_heavy_line)
    cv2.line(img, (x1, y1), (x1 + offset, y1 ), color, thickness_heavy_line)
    
    # Left Bottom (x1, y2)
    cv2.line(img, (x1, y2), (x1, y2 - offset), color, thickness_heavy_line)
    cv2.line(img, (x1, y2), (x1 + offset, y2 ), color, thickness_heavy_line)

    # Right Top (x2, y1)
    cv2.line(img, (x2, y1), (x2, y1 + offset), color, thickness_heavy_line)
    cv2.line(img, (x2, y1), (x2 - offset, y1), color, thickness_heavy_line)

    # Right Bottom (x2, y2)
    cv2.line(img, (x2, y2), (x2, y2 - offset), color, thickness_heavy_line)
    cv2.line(img, (x2, y2), (x2 - offset, y2 ), color, thickness_heavy_line)
    
    # Draw long line
    cv2.line(img, (x1, y1), (x1, y2), color, thickness_slim_line)
    cv2.line(img, (x1, y1), (x2, y1), color, thickness_slim_line)
    cv2.line(img, (x2, y2), (x1, y2), color, thickness_slim_line)
    cv2.line(img, (x2, y2), (x2, y1), color, thickness_slim_line)
    return img


def load_pretrain_model(is_printed=False):
    ''' Load pretrain model saved as json file and its weights
        Configurated model path in gaconfig.py
        
        Parameter(s):
            is_printed: Show architecture of Model in console/terminal
    '''
    with open(agconfig.MODEL_ARCHITECTURE_JSON, 'r') as file:
        json_arc = file.read()
        model = model_from_json(json_arc)
        model.load_weights(agconfig.WEIGHT_PATH)

        if is_printed:
            print(model.summary())

        return model


# Havent TEST YET
def save_trained_model(model, path):
    
    ''' Load pretrain model saved as json file and its weights
        Configurated model path in gaconfig.py
        
        Parameter(s):
            is_printed: Show architecture of Model in console/terminal
    '''
    with open(path, 'w') as file:
        # json_arc = file.read()
        model.to_json(file)
        model.save_weights(agconfig.WEIGHT_PATH)
        return model


def resize_with_ratio(PIL_img, screen_w, screen_h):
    width, height = PIL_img.size
    ratio = width/height
    # Resize image height and keep ratio
    if width > height:
        return PIL_img.resize((int(screen_h*ratio), screen_h), Image.ANTIALIAS)
    else:
        return PIL_img.resize((screen_w, int(screen_w*ratio)), Image.ANTIALIAS)


def preprocess_image_for_test(img):  
    face_rect_resized = cv2.resize(img, (64, 64))
    face_rect_reshape = np.reshape(face_rect_resized, newshape=(1, 64, 64, 3))
    face_rect_reshape = preprocess_image(face_rect_reshape)
    return face_rect_reshape

        