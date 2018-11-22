import cv2
import numpy as np

def preprocessing_function(x):
    x = padding_image(x)
    x = resize(x, target_size=(128, 128))
    x = normalize(x)
    return x


def resize(x, target_size=(128, 128)):
    return cv2.resize(x, target_size)

    
def padding_image(x, padding_type = 'nearest'):
    """Preprocesses a numpy array encoding a batch of images.

    This function applies the "Inception" preprocessing which converts
    the RGB values from [0, 255] to [-1, 1]. Note that this preprocessing
    function is different from `imagenet_utils.preprocess_input()`.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
        padding_type: 
                    - default: zero padding
                    - white:   white padding (padding using values at 255)
                    - nearest: pad using nearest pixel intensity
    # Returns
        Preprocessed array.
    """
    old_h, old_w, _ = x.shape
    delta = 0
    top, bottom, left, right = 0, 0, 0, 0
    
    if old_w > old_h:
        delta = old_w - old_h
        top, bottom = delta//2, delta - (delta//2)
    else:
        delta = old_h - old_w
        left, right = delta//2, delta - (delta//2)
    color = [0, 0, 0]
    
    return cv2.copyMakeBorder(x, top, bottom, left, right, cv2.BORDER_REPLICATE, value=color)


def normalize(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    x = x/255.0
    return x
