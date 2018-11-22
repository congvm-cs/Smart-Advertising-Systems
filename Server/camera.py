import cv2
# local modules
from src.multi_tracking import MultiTracking

# Init
mtking = MultiTracking()

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        print("LOAD VIDEO CAPTURE")
        self.video = cv2.VideoCapture(0)            


    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        _, frame = self.video.read()

        image = mtking.detectAndTrackMultipleFaces(1, frame)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    # def get_byte_frame(self):
    #     _, image = self.video.read()
    #     image = self.qrscanner.scan_return_image(image)
    #     # We are using Motion JPEG, but OpenCV defaults to capture raw images,
    #     # so we must encode it into JPEG in order to correctly display the
    #     # video stream.
    #     _, jpeg = cv2.imencode('.jpg', image)
    #     return jpeg.tobytes()