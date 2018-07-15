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

# Init
app = Flask(__name__)
mtking = MultiTracking()
information = {}

@app.route('/upload_data', methods=['POST'])
def post_inform():
    if request.method == 'POST':
        content = request.json
        # image = content['image'][2:-1]    # get string format only
        image = content['image']

        print(image[:5])
        # Decode
        pad = len(image) % 4            # length of encoded code must be multiply of 4
        # print(pad) 

        image += "="*pad

        pad = len(image) % 4            # length of encoded code must be multiply of 4
        print(pad) 

        image = image.encode('ascii')        # convert to byte
        image_decoded = base64.b64decode(image)     # decode 
        PIL_img = Image.frombuffer('RGB', (224, 168), 
                                    image_decoded, 'raw', 'RGB', 0, 1)  # convert to image
        img = np.array(PIL_img)
        
        # Tracking
        mtking.detectAndTrackMultipleFaces(img)

    return Response(content)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000)

