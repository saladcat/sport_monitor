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
        plt.figure(figsize=(4, 3))
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
                np.asarray(self.time_seq_dict[index]).reshape(-1, 1),
                np.asarray(self.std_seq_dict[index]).reshape(-1, 1),
                dist=lambda x, y: np.linalg.norm(x - y, ord=1))
            # dist=lambda x, y: math.sqrt((x * x) + (y * y)))
            # dist=lambda x, y: math.fabs((x -y)))

            sum_dist += dist
            score["index"] = (dist, cost, acc, path)

        return sum_dist, score

    def save_as_std(self, file_name="std_dump_file"):
        mydb = open(file_name, 'w')
        pickle.dump(self.time_seq_dict, mydb)
