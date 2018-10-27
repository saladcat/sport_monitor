from SitupPose import SitupPose
from BasketPose import BasketPose
from PushupPose import PushupPose
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

if __name__ == '__main__':
    # a = SitupPose("/home/zhu/Desktop/1.gif")
    a = BasketPose("/home/zhu/Desktop/2.gif")
    # a = PushupPose("/home/zhu/Desktop/3.gif")
    ret = True
    while ret is True:
        ret, img, prams = a.process()
        if ret is False:
            break
        cv2.imshow("output", img)
        cv2.waitKey(1)
    print a.get_time_seq_pram()
    print a.eval()[0]
