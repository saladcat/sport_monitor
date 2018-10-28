import numpy as np
from PoseHelper import get_distance, get_angle, rad2ang
from Pose import Pose
import cv2
import matplotlib.pyplot as plt
import pickle


class SitupPose(Pose):
    def __init__(self, video_name):
        super(SitupPose, self).__init__(video_name)
        self.time_seq_dict = {
            "head_angle": [],
            "body_angle": [],
        }

    def process(self):
        ret, img = self.video.read()
        if ret is True:
            prams = {
                "head_angle": 0,
                "body_angle": 0
            }
            keypoints, output_img = self.openpose.forward(img, True)

            prams["head_angle"], prams["body_angle"] = get_situp_angles(keypoints[0])

            cv2.putText(output_img,
                        "head_angle=%f,body_angle=%f" % (rad2ang(prams["head_angle"]), rad2ang(prams["body_angle"])),
                        (10, 20),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 0, 255), 1)
            self._record(prams)
            output_img = cv2.resize(output_img, (400, 300), interpolation=cv2.INTER_CUBIC)
        else:
            output_img = None
            prams = None

        return ret, output_img, prams

    def eval(self, file_name="resource/std_situp"):
        return super(SitupPose, self).eval(file_name)

    def get_time_seq_pram(self, file_name="output/situp.jpg"):
        return super(SitupPose, self).get_time_seq_pram(file_name)


def get_situp_angles(keypoint):
    head_ang, head_confidence = get_angle(keypoint, 0, 1, 8)
    body_ang, body_confidence = get_angle(keypoint, 1, 8, 10)
    return head_ang, body_ang
