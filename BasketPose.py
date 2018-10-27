import numpy as np
from PoseHelper import get_distance, get_angle, rad2ang
from Pose import Pose
import cv2
import matplotlib.pyplot as plt


class BasketPose(Pose):
    def __init__(self, video_name):
        super(BasketPose, self).__init__(video_name)
        self.time_seq_dict = {
            "arm1": [],
            "arm2": [],
            "body": [],
        }

    def process(self):
        ret, img = self.video.read()
        if ret is True:
            prams = {
                "arm1": 0,
                "arm2": 0,
                "body": 0,
            }
            keypoints, output_img = self.openpose.forward(img, True)

            prams["arm1"], prams["arm2"], prams["body"] = get_basket_angles(keypoints[0])

            cv2.putText(output_img,
                        "arm1=%f,arm2=%f,body=%f" % (
                            rad2ang(prams["arm1"]), rad2ang(prams["arm2"]), rad2ang(prams["body"])),
                        (10, 20),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.3,
                        (0, 0, 255), 1)
            self._record(prams)
        else:
            output_img = None
            prams = None
        return ret, output_img, prams

    def eval(self, file_name="resource/std_basketball"):
        return super(BasketPose, self).eval(file_name)


def get_basket_angles(keypoint):
    angel_arm1_right, confidence_arm1_right = get_angle(keypoint, 2, 3, 4)  # xiao ge bo & da ge bo
    angel_arm2_right, confidence_arm2_right = get_angle(keypoint, 1, 2, 3)  # da ge bo & jian bang
    return angel_arm1_right, angel_arm2_right, angel_arm3_right
