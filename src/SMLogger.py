import sys
sys.path.append('..')
from src.config import config
import time
import os

class SMLogger():
    def __init__(self):
        self.f = None

    def update(self, dic_result):
        # self.dic_result['video_index'] = 
        pass

    def open(self, filename='Log.LOG'):
        self.f = open(file=filename, mode='w')

    def write(self, startTime, videoIndex, dic_results):
        endTime = time.localtime()
        dic_results['startTime'] = list(startTime)[0:6]
        dic_results['endTime'] = list(endTime)[0:6]
        dic_results['videoIndex'] = videoIndex
        content = '{}\n'.format(dic_results)
        self.f.write(content)


    def close(self):
        self.f.close()
