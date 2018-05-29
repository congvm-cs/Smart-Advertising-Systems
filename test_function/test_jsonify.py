import numpy as np 
import codecs, json
import cv2 

image = cv2.imread('/Users/ngocphu/Documents/FINAL_PROJECT_RESEARCH/flask_streaming/NTP_4866.JPG',0)

image = cv2.resize(image, (128,128))
print(type(image))
print(image.shape)
a = image.tolist()
# print(a.shape, '\n' ,type(a))
file_path = "path.json"
json.dump(a, codecs.open(file_path, 'wb', encoding='utf-8'), separators= (',',':'), sort_keys=True, indent=4)