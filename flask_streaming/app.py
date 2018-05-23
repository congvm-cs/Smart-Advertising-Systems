from flask import Flask, render_template, Response, request
import cv2

app = Flask(__name__)
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
        # print(content['image'])
        print(type(information))
    return Response(content)


if __name__ == '__main__':
    app.run(debug=True)

