import cv2
import PoseHelper
from PoseHelper import params, OpenPose
import matplotlib.pyplot as plt
from dtw import dtw
import math
import pickle
import numpy as np


class Pose(object):
    def __init__(self, video_name):
        self.video = cv2.VideoCapture(video_name)
        self.openpose = OpenPose(params)
        self.std_seq_dict = {}
        self.time_seq_dict = {}

    def process(self):
        pass

    def _record(self, dict):
        for index in dict:
            self.time_seq_dict[index].append(dict[index])

    def get_time_seq_pram(self, file_name):
        for index in self.time_seq_dict:
            x = [i for i in range(len(self.time_seq_dict[index]))]
            plt.plot(x, self.time_seq_dict[index], label=index)
        plt.legend()
        plt.savefig(file_name)
        plt.show()

        return self.time_seq_dict

    def eval(self, file_name):
        mydb = open(file_name, 'r')
        self.std_seq_dict = pickle.load(mydb)
        score = {}
        sum_dist = 0
        for index in self.time_seq_dict:
            dist, cost, acc, path = dtw(
                self.time_seq_dict[index],
                self.std_seq_dict[index],
                dist=lambda x, y: math.sqrt((x * x) + (y * y)))
            sum_dist += dist
            score["index"] = (dist, cost, acc, path)

        return sum_dist, score
