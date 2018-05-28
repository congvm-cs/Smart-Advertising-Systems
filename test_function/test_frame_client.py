import cv2
import requests
import codecs, json
import time
import flask
import base64

image_arr = []
information = {}
def video_feed():
    cap = cv2.VideoCapture(0)
    # cap.set(CV_CAP_PROP_FPS, 20)
    while True: 
        fake_json = []
        start = time.time()
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (128,128))
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.resize(gray, (128,128))
        if not ret:
            print("Error: failed to capture image")
            break

        encoded_img = base64.b64encode(frame)
        # image_arr.append(frame)
        #cv2.imshow('gray', gray)
        # a = frame.tobytes()
        # file_path = "path.json"
        # print(a)
        # cv2.waitKey(50)
        information['image'] = str(encoded_img)
        json.dumps(information)
        # fake_json.append(information.copy())
        # flask.jsonify(information)
        # real_json = flask.jsonify(fake_json)
        # print(fake_json)
        end = time.time()
        # cv2.imwrite('now.jpeg', gray)
        print(end-start)
    cap.release()
    cv2.destroyAllWindows()

    # np.save('image_mat', image_arr)
    # upload = requests.post()
video_feed()

        
