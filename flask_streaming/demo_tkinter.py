import sys
PYTHON_VERSION = sys.version_info[0]

if PYTHON_VERSION == 2:
    import Tkinter as tk
    from ttk import Frame, Button
else:
    import tkinter as tk
    from tkinter import Frame, Button

from PIL import Image
from PIL import ImageTk
import cv2
import time
import numpy as np
import skvideo.io
import skvideo.datasets
import queue
import threading
        

class App:
    def __init__(self, window_title, video_source):

        self.workQueue = queue.Queue()
        # (132, 720, 1280, 3)
        self.vid = skvideo.io.vreader(video_source)
        for frame in self.vid:
            self.width = frame.shape[1]
            self.height = frame.shape[0]
            
            break
        
        self.window = tk.Tk()
        self.window.title(window_title)
        # self.window.overrideredirect(True)
        # self.window.overrideredirect(False)
        self.window.attributes('-fullscreen', True)
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()
        self.window.geometry('{}x{}+{}+{}'.format(self.screen_width, self.screen_height, 0, 0))

        print(self.width)
        print(self.height)
        print('__init__')
        
        self.canvas = tk.Canvas(self.window, width = self.width, height = self.height)
        self.canvas.pack()
        
        # self.panel = tk.Label(self.window, image=self.photo)
        # self.panel.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.thread_2 = threading.Thread(target=self.__read_images)
        self.thread_1 = threading.Thread(target=self.update)
        self.thread_2.start()
        self.thread_1.start()
        self.delay = 1
        # self.update()
        self.window.mainloop()
            
    
    def update(self):
        # Get a frame from the video source
        #ret, frame = self.vid.get_frame()
        
        # self.panel = tk.Label(self.window, image=self.photo)
        while not self.workQueue.empty():
            self.photo = self.workQueue.get()
        self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.window.after(self.delay, self.update)
        print ('1')

    def __read_images(self):
        for frame in self.vid:
            self.image = Image.fromarray(frame)
            self.image = self.image.resize((self.screen_width, self.screen_height), Image.ANTIALIAS)
            self.workQueue.put(self.image)
            # break
        print ('2')
        
App("Tkinter and OpenCV","/Users/ngocphu/Documents/FINAL_PROJECT_RESEARCH/Smart_Advertising_Systems/Smart-Advertising-Systems/flask_streaming/1.avi")
