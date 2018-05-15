#!/usr/bin/python

import sys
sys.path.append('./Models')
# for multi-processing
import threading
from multiprocessing import Process

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
from Models.agutils import resize_with_ratio

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

        self.root.after(3000, self.__update_image)
        self.root.mainloop()


    def __load_images(self, image_paths):
        return glob.glob(image_paths)


    def __read_images(self, single_image_path):
        image = Image.open(single_image_path)
        
        image = resize_with_ratio(image, self.screen_width, self.screen_height)
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
        self.root.after(3000, self.__update_image)       # Set to call again in 30 seconds

        # TODO here



def show():
    # cap = cv2.VideoCapture(0)

    # while True:
    #     ret, frame = cap.read()

    #     cv2.imshow('hello', frame)
    #     cv2.waitKey(1)

    pass


def main():
    image_paths = './Ads_images/*/*.jpg'
    # show_camera = Process(target=show)
    # show_camera.start()

    show_ads = Process(target=SmartAds, args=[image_paths])
    show_ads.start() 
    

if __name__ == '__main__':
    main()
