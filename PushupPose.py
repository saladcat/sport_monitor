import numpy as np
from PoseHelper import get_distance, get_angle, rad2ang
from Pose import Pose
import cv2
import matplotlib.pyplot as plt


class PushupPose(Pose):
    def __init__(self, video_name):
        super(PushupPose, self).__init__(video_name)
        self.time_seq_dict = {
            "body": [],
            "arm": [],
        }

    def process(self):
        ret, img = self.video.read()
        if ret is True:
            prams = {
                "body": 0,
                "arm": 0,
            }
            keypoints, output_img = self.openpose.forward(img, True)

            prams["body"], prams["arm"] = get_pushup_angles(keypoints[0])

            cv2.putText(output_img,
                        "body=%f,arm=%f" % (
                            rad2ang(prams["body"]), rad2ang(prams["arm"])),
                        (10, 20),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (0, 0, 255), 1)
            self._record(prams)
        else:
            output_img = None
            prams = None
        return ret, output_img, prams

    def eval(self, file_name="resource/std_pushup"):
        return super(PushupPose, self).eval(file_name)

    def get_time_seq_pram(self, file_name="output/pushup.jpg"):
        return super(PushupPose, self).get_time_seq_pram(file_name)


def get_pushup_angles(keypoint):
    angel_body_right, confidence_body_right = get_angle(keypoint, 1, 8, 13)  # xiao ge bo & da ge bo
    angel_arm_right, confidence_arm_right = get_angle(keypoint, 5, 6, 7)  # da ge bo & jian bang
    return angel_body_right, angel_arm_right
