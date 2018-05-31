#!/usr/bin/python

# System modules
import sys
sys.path.append('..')
import os
import numpy as np
from PIL import Image
from PIL import ImageTk
import time
import glob
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    import Tkinter as tk
    from ttk import Frame, Button
else:
    import tkinter as tk
    from tkinter import Frame, Button
import cv2
from threading import Thread

# Local modules
# from src.utils import resize_with_ratio

import skvideo.io
# from Client import Client
import omxplayer

VIDEO_DIR = '/Users/ngocphu/Smart-Advertising-Systems/Ad_videos'
VIDEO_PATH = []
class SmartAds():
    def __init__(self):
        # Image current index to show
        self.list_videos()
        # Client using webcam
        # self.clt = Client()
        
        for video in VIDEO_PATH:
            self.idx = self.get_index(video)
            self.player = omxplayer.OMXPlayer(video, pause=True)
        # self.player.set_aspect_mode('fill')
    
    def start(self):
        self.player.play()
        # self.clt.start()

    def stop(self):
        self.player.quit()

    def list_videos(self):
        
        for video_file in os.listdir(VIDEO_DIR):
            if video_file != '.DS_Store':
                self.video_path = os.path.join(VIDEO_DIR,video_file)
                VIDEO_PATH.append(self.video_path)

    def get_index(self, video_name):
        idx = video_name[:1]
        return idx

def main():
    # file_paths = '/home/pi/1.avi'
    sa = SmartAds()	
    sa.start()
#	time.sleep(10)
#	sa.stop()

if __name__ == '__main__':
    main()
