#!/usr/bin/python

# System modules
import sys
sys.path.append('./Models')
import numpy as np
import queue
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


# Local modules
from Models.agutils import resize_with_ratio
import skvideo.io
from threading import Thread


class SmartAds():
    def __init__(self, video_source):
        # Image current index to show
        self.index = 0
        self.vid = skvideo.io.vreader(video_source)
        self.t = time.time()

        self.root = tk.Tk()
        self.root.title('Smart Ads System')

        self.q = queue.Queue(32)
        self.stopped = False

        # make app be in fullscreen mode
        self.root.overrideredirect(True)
        self.root.overrideredirect(False)
        self.root.attributes('-fullscreen', True)
        
        # get the image size
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # make the root window the size of the image
        # self.root.geometry('{}x{}+{}+{}'.format(self.screen_width, self.screen_height, 0, 0))
                
        # Use a label as a panel
        self.panel = tk.Label(self.root)
        self.panel.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.panel.pack()


    def start(self):
        # Start buffer
        self.__start_buffer()
        time.sleep(1)

        print('Init done!')
        self.__update_frame()
        self.root.mainloop()


    def stop(self):
        self.stopped = True

    
    def __start_buffer(self):
        print('Start buffer!')
		# start a thread to read frames from the file video stream
        t = Thread(target=self.__add_frame_into_buffer, args=())
        t.daemon = True
        t.start()


    def __add_frame_into_buffer(self):
        print('Add frame into buffer!')
        # keep looping infinitely
        while True:
            time.sleep(0.001)
			# if the thread indicator variable is set, stop the
			# thread
            if self.stopped:
                return

            # otherwise, ensure the queue has room in it
            if not self.q.full():
                # read the next frame from the file
                try:
                    frame = next(self.vid)
                    frame = Image.fromarray(frame)
                    # frame = resize_with_ratio(frame, self.screen_width, self.screen_height)
                    frame = frame.resize((self.screen_width, self.screen_height), Image.ANTIALIAS)
                    # add the frame to the queue
                    self.q.put(frame)

                except StopIteration:
                    self.stopped = True
                    print('Loaded all video into buffer')
                
                
    def check_size_buffer(self):
		# return True if there are still frames in the queue
        return self.q.qsize() > 0


    def __load_images(self, image_paths):
        return glob.glob(image_paths)


    def __get_frame(self):
        _frame = self.q.get(block=False, timeout=2.0)
        return ImageTk.PhotoImage(_frame)


    def __update_frame(self):
        '''This function to show Images consequencely
        '''
        if self.check_size_buffer():
            self.index += 1

            if np.round(time.time() - self.t) == 1:
                print('fps: {}'.format(self.index))
                self.index = 0
                self.t = time.time()

            self.image = self.__get_frame()
            self.panel.configure(image=self.image)
            self.panel.image = self.image
            self.root.after(1, self.__update_frame)


def main():
    file_paths = 'C:\\Users\\VMC\\Desktop\\1.avi'
    sa = SmartAds(file_paths)
    sa.start()


if __name__ == '__main__':
    main()