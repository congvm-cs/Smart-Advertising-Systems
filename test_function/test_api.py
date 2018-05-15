# import base64 
import requests as r 

# with open("/Users/ngocphu/Documents/FINAL_PROJECT_RESEARCH/NTP_4866.JPG", "rb") as image_file:
#     encoded = base64.b64encode(image_file.read())

file = {'image': open('/Users/ngocphu/Documents/FINAL_PROJECT_RESEARCH/NTP_4866.JPG', 'rb') }
print(type(file))
upload = r.post("http://127.0.0.1:8000/task/", files = file)
# print(type(encoded))
print(upload.text)