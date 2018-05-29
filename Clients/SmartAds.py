#!/usr/bin/python

# System modules
import sys
sys.path.append('..')

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
import omxplayer


class SmartAds():
	def __init__(self, video_source):
		# Image current index to show

		# Client using webcam
		self.clt = Client()
		self.player = omxplayer.OMXPlayer(video_source, pause=True)
		# self.player.set_aspect_mode('fill')

	def start(self):
		self.player.play()
		self.clt.start()



	def stop(self):
		self.player.quit()



def main():
    	file_paths = '/home/pi/1.avi'
    	sa = SmartAds(file_paths)	
    	sa.start()
#	time.sleep(10)
#	sa.stop()

if __name__ == '__main__':
    main()
