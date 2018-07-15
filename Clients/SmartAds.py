#!/usr/bin/python

# System modules
import sys
sys.path.append('..')
import os
import numpy as np
import time
import glob
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    import Tkinter as tk
    from ttk import Frame, Button
else:
    import tkinter as tk
    from tkinter import Frame, Button, Label
import cv2
from threading import Thread

# Local modules
# from src.utils import resize_with_ratio

from Client import Client
import omxplayer
import subprocess
from subprocess import PIPE
import sys
import signal
VIDEO_DIR = '/Users/ngocphu/Smart-Advertising-Systems/Ad_videos'
VIDEO_PATH = []
class SmartAds():
    def __init__(self, loop, play_list):
        # Image current index to show
        #lf.list_videos()
        # Client using webcam
        self.root = tk.Tk()
        self.root.title('Smart Ad')
        self.root.overrideredirect(True)
        self.root.overrideredirect(False)
        self.root.attributes('-fullscreen', True)
        self.panel = tk.Label(self.root)
        self.panel.config(fg = 'black', bg='black')
        self.panel.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.clt = Client()
        self.index = 0
        self.loops = loop
        self.play_list = play_list
        self.current_loop = 0 
        self.clt.start()
        self.run()     
        self.root.mainloop()

    def run(self):
        if self.current_loop < self.loops:
            self.clt.set_index(self.index)
            p1 = subprocess.call(['omxplayer', '-b', self.play_list[self.index], 'daemon'], stdout= PIPE)
        print('3')

        if (self.index + 1) >= len(self.play_list):
            self.index = 0
            self.current_loop += 1
        else:
            self.index += 1

        self.root.after(1, func=self.run)


    def set_index(self, index):
        self.clt.set_index(index)


def main():
    # file_paths = '/home/pi/1.avi'
    try:
        loops = int(input("> How many loops? "))
        play_list = sorted(glob.glob('/home/pi/Smart-Advertising-Systems/Ad_Videos/*.avi'))
        sa = SmartAds(loops, play_list)
        
    except KeyboardInterrupt:
        print('stop')
    
    sys.exit()
            
            
if __name__ == '__main__':
    main()
