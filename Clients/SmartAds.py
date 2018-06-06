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
from Client import Client
import vlc
import glob
import omxplayer
import subprocess
from subprocess import PIPE
import sys

VIDEO_DIR = '/Users/ngocphu/Smart-Advertising-Systems/Ad_videos'
VIDEO_PATH = []
class SmartAds():
    def __init__(self):
        # Image current index to show
        #lf.list_videos()
        # Client using webcam
        self.clt = Client()
	index = 0       
	# load playlist
	
	
	#print(self.play_list)
        #self.player = vlc.Instance('--input-repeat=-1', '--fullscreen', '--mouse-hide-timeout=0')

        #self.player = omxplayer.OMXPlayer(self.play_list[self.index], pause=False)
        #self.player.set_aspect_mode('fill')


    def run(self):
	self.clt.start()

	time.sleep(2)
	print('2')
	#self.player.load(self.play_list[1])	

"""
    def list_videos(self):
        
        for video_file in os.listdir(VIDEO_DIR):
            if video_file != '.DS_Store':
                self.video_path = os.path.join(VIDEO_DIR,video_file)
                VIDEO_PATH.append(self.video_path)

    def get_index(self, video_name):
        idx = video_name[:1]
        return idx

"""
def main():
    # file_paths = '/home/pi/1.avi'
    play_list = sorted(glob.glob('/home/pi/Smart-Advertising-Systems/Ad_Videos/*.avi'))
    sa = SmartAds()
    sa.run()

    while True:
        for idx, video in enumerate(play_list):
            #index = video.index
            p1 = subprocess.Popen(['omxplayer', '-b', video], stdout= PIPE)
            p1.wait()
"""
    if sa.isPlaying():
	print('1')
	sa.next()
	time.sleep(2)
"""
if __name__ == '__main__':
    main()
