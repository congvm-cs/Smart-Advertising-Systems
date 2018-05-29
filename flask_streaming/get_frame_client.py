import requests 
from cv2 import VideoCapture, resize
import base64 
import json
import time

cap = VideoCapture(1)
information = {}

while True:
    information.clear()
    _, image = cap.read()
    image = resize(image, (0, 0), fx=0.4, fy=0.4)

    print(image.shape)
    image_encoded = base64.b64encode(image)
    # image_encoded = image.tostring()
    # print(type(image_encoded))
    # print(image_encoded)
    information['ad_index'] = 1
    information['image'] = str(image_encoded)

    # json_data = json.dumps(information)
    # print(information)
    # print (type(information), type(json_data))
    # fake_dick.append(information)
    posted = requests.post('http://127.0.0.1:5000/upload_data', json = information)
