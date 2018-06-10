import requests 
from cv2 import VideoCapture
import base64 
import json
import time

cap = VideoCapture(0)
information = {}

while True:
    information.clear()
    _, image = cap.read()
    # image = cv2.resize(image, (0, 0), fx=1, fy=1)
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
