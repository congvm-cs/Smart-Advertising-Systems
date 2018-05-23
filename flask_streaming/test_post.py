import requests 
import cv2
import base64 
import json
import time
from multiprocessing import Pool

# cap = cv2.VideoCapture(0)
information = {}
while True:
    start = time.time()
    image = cv2.imread('/Users/ngocphu/Documents/FINAL_PROJECT_RESEARCH/flask_streaming/NTP_4866.JPG')

    # _,image = cap.read()
    image = cv2.resize(image, (640,320))
    
    # _, image_encoded = cv2.imencode('.jpg', image)
    # image_encoded = base64.b64encode(image)
    image_encoded = image.tostring()
    # print(type(image_encoded))
    # print(image_encoded)
    information['ad_index'] = 1
    information['image'] = str(image_encoded)

    # json_data = json.dumps(information)
    # print(information)
    # print (type(information), type(json_data))
    # fake_dict.append(information)
    posted = requests.post('http://127.0.0.1:5000/upload_data', json = information)
    end = time.time()
    print(end-start)
    # return(end-start)
if __name__ == '__main__':
    term

# print(posted.text)
# print (posted.content)