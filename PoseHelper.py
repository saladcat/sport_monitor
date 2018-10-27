import numpy as np
import cv2
import math
# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform

# Remember to add your installation path here
# Option a
if platform == "win32":
    sys.path.append(dir_path + '/../../python/openpose/');
else:
    sys.path.append('../../python');
# Option b
# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
# sys.path.append('/usr/local/python')

# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
try:
    from openpose import *
except:
    raise Exception(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is bdir_path = os.path.dirname(os.path.realpath(__file__))uilt, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = dir_path + "/../../../models/"


# Construct OpenPose object allocates GPU memory

def get_distance(keypoints, shoes_size=42, threshold=0.6):
    LHS = np.r_[keypoints[6], keypoints[7]].reshape(2, 3)
    RHS = np.r_[keypoints[3], keypoints[4]].reshape(2, 3)
    use_LHS = True
    use_RHS = True
    for i in range(2):
        if LHS[i][2] < threshold:
            use_LHS = False
        if RHS[i][2] < threshold:
            use_RHS = False
    if not use_RHS and not use_LHS:
        return False, 0
    else:
        leg = RHS if use_RHS is True else LHS

    pix_dis = _l2_distance(leg[0], leg[1])
    real_dis = _shoes_size_to_cm(shoes_size)

    return True, real_dis / pix_dis


def get_angle(keypoint, p1, p2, p3):
    point1 = keypoint[p1]
    point2 = keypoint[p2]
    point3 = keypoint[p3]
    l1 = math.sqrt(
        (point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]))
    l2 = math.sqrt(
        (point2[0] - point3[0]) * (point2[0] - point3[0]) + (point2[1] - point3[1]) * (point2[1] - point3[1]))
    l3 = math.sqrt(
        (point3[0] - point1[0]) * (point3[0] - point1[0]) + (point3[1] - point1[1]) * (point3[1] - point1[1]))
    result = math.acos((l1 * l1 + l2 * l2 - l3 * l3) / (2 * l1 * l2 + 1e-5))
    confidence = (point1[2], point2[2], point3[2])
    return result, confidence


def _shoes_size_to_cm(shoes_size):
    cm_size = (shoes_size + 10) / 2
    return cm_size


def _l2_distance(x, y):
    return np.sqrt((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]))


def rad2ang(rad):
    return rad / math.pi * 180
