#!/usr/bin/python

# use a Tkinter label as a panel/frame with a background image
# note that Tkinter only reads gif and ppm images
# use the Python Image Library (PIL) for other image formats
# free from [url]http://www.pythonware.com/products/pil/index.htm[/url]
# give Tkinter a namespace to avoid conflicts with PIL
# (they both have a class named Image)
import sys
import threading
from mtcnn.mtcnn import MTCNN
import numpy as np


PYTHON_VERSION = sys.version_info[0]

if PYTHON_VERSION == 2:
    import Tkinter as tk
    from ttk import Frame, Button
else:
    import tkinter as tk
    from tkinter import Frame, Button

from PIL import Image
from PIL import ImageTk

import time
import glob

import cv2


class SmartAds():
    def __init__(self, image_paths):
        
        # Image current index to show
        self.index = 0

        self.root = tk.Tk()
        self.root.title('Smart Ads System')

        # make app be in fullscreen mode
        self.root.overrideredirect(True)
        self.root.overrideredirect(False)
        self.root.attributes('-fullscreen', True)
        
        # get the image size
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # make the root window the size of the image
        self.root.geometry('{}x{}+{}+{}'.format(self.screen_width, self.screen_height, 0, 0))

        # pick an image file you have .bmp  .jpg  .gif.  .png
        # load the file and covert it to a Tkinter image object
        self.image_paths = self.__load_images(image_paths)
        print('Length of image array: ', len(self.image_paths))

        # Read images
        self.current_image = self.__read_images(self.image_paths[self.index])
                
        # Use a label as a panel
        self.panel = tk.Label(self.root, image=self.current_image)
        self.panel.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        print("Display image {}".format(self.index))

        self.root.after(1000, self.__update_image)
        self.root.mainloop()


    def __load_images(self, image_paths):
        return glob.glob(image_paths)


    def __read_images(self, single_image_path):
        image = Image.open(single_image_path)
        image = image.resize((self.screen_width, self.screen_height), Image.ANTIALIAS)
        return ImageTk.PhotoImage(image)


    def __update_image(self):
        '''This function to show Images consequencely
        '''
        if (self.index + 1) <= (len(self.image_paths) - 1):
            self.index += 1  
        else:
            self.index = 0

        print("Display image", self.index)
        self.current_image = self.__read_images(self.image_paths[self.index])
        # self.next_image = self.__read_images(self.image_paths[self.index + 1])  if (self.index + 1) <= len(self.image_paths) else 0
        
        # if self.display == self.current_image:
        self.panel.configure(image=self.current_image)
        self.root.after(1000, self.__update_image)       # Set to call again in 30 seconds

        # TODO here



def show():
    cap = cv2.VideoCapture(0)
    t = 0
    fps = 0
    fps_print = 0

    detector = MTCNN()
    ad = 0.4

    while True:
        # t+=1
        #  # Start timer
        # timer = cv2.getTickCount()

        _, frame = cap.read()
        # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # img_h, img_w, _ = np.shape(frame)

        # # Display FPS on frame
        
        
        # if t % 10 == 0:
        #     fps_print = fps/40
        #     fps = 0
        #     t = 0

        # cv2.putText(frame, "FPS : " + str(int(fps_print)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        # detected = detector.detect_faces(frame)
        #     # faces = np.empty((len(detected), img_size, img_size, 3))

        # for i, d in enumerate(detected):
        #     print(i)
        #     print(d['confidence'])
        #     if d['confidence'] > 0.95:
        #         x1, y1, w, h = d['box']
        #         x2 = x1 + w
        #         y2 = y1 + h
        #         xw1 = max(int(x1 - ad * w), 0)
        #         yw1 = max(int(y1 - ad * h), 0)
        #         xw2 = min(int(x2 + ad * w), img_w - 1)
        #         yw2 = min(int(y2 + ad * h), img_h - 1)
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # fps += cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.imshow('hello', frame)
        cv2.waitKey(1)


def main():
    image_paths = './Ads_images/*/*.jpg'
    show_camera = threading.Thread(target=show)
    show_camera.start()

    show_ads = threading.Thread(target=SmartAds, args=[image_paths])
    show_ads.start()

if __name__ == '__main__':
    main()
