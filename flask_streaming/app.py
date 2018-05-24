import sys
sys.path.append('..')
from Models.multi_tracking_dlib import MultiTracking 
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import base64
from PIL import Image


app = Flask(__name__)
mtking = MultiTracking()

information = {}

@app.route('/')
def index():
    return render_template('index.html')


def gen():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to capture image")
            break

        cv2.imwrite('demo.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')


@app.route('/connected', methods=['GET'])
def connected():
    if request.method == 'GET': 
        return Response('Connected')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload_data', methods=['POST'])
def post_inform():
    if request.method == 'POST':
        # information['ad_index'] = request.form['ad_index']
        # information['image'] = request.form['image']
        content = request.json
        # print(content['ad_index'])
        # print('---------------------------------------------------')

        image = content['image'][2:-1]    # get string format only
        
        # Decode
        pad = len(image) % 4            # length of encoded code must be multiply of 4
        image += "="*pad        
        image = image.encode()          # convert to byte
        image_decoded = base64.b64decode(image)     # decode 
        PIL_img = Image.frombuffer('RGB', (640, 480), 
                                    image_decoded, 'raw', 'RGB', 0, 1)  # convert to image
        img = np.array(PIL_img)
        
        # Tracking
        mtking.detectAndTrackMultipleFaces(img)

    return Response(content)

if __name__ == '__main__':
    app.run(debug=False)

