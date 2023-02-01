#!/usr/bin/env python3
from __future__ import print_function

from numpy import dtype
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_pytorch import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from nav_msgs.msg import Path
from std_msgs.msg import Int8MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse
import csv
import os
import time
import copy
import sys
import tf
from nav_msgs.msg import Odometry

class theta_to_3camera:
    def __init__(self):
        rospy.init_node('theta_to_3camera', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/image_raw", Image, self.callback)
        self.theta_image = np.zeros((720,1280,3), np.uint8)
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.learning = True
        self.select_dl = False

    def callback(self, data):
        try:
            self.theta_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
    def loop(self):
        if self.theta_image.size != 1280 * 720 * 3:
            return
        
        self.cv_image = self.theta_image[120 : 599 , 320 : 959]
        img = resize(self.cv_image, (48, 64), mode='constant')
        self.cv_left_image = self.theta_image[120 : 599 , 280 : 919]
        img_left = resize(self.cv_left_image, (48, 64), mode='constant')
        self.cv_right_image = self.theta_image[120 : 599 , 360 : 999]
        img_right = resize(self.cv_right_image, (48, 64), mode='constant')

        temp = copy.deepcopy(img)
        cv2.imshow("Resized Image", temp)
        temp = copy.deepcopy(img_left)
        cv2.imshow("Resized Left Image", temp)
        temp = copy.deepcopy(img_right)
        cv2.imshow("Resized Right Image", temp)
        cv2.waitKey(1)

if __name__ == '__main__':
    t3 = theta_to_3camera()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        t3.loop()
        r.sleep()
        



