# system modules
import sys
sys.path.append('..')

import cv2
import numpy as np
import base64
from PIL import Image
from flask import Flask, render_template, Response, request

# local modules
from src.multi_tracking import MultiTracking
from camera import VideoCamera

# Init
app = Flask(__name__)
mtking = MultiTracking()

cap = VideoCamera()



@app.route('/video')
def video():
    return app.send_static_file('dienmayxanh.mp4')


@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(cap),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=13000, debug=False)
    

