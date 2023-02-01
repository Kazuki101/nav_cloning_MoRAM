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
import numpy as np

class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        self.mode = rospy.get_param("/nav_cloning_node/mode", "use_dl_output")
        self.action_num = 1
        self.dl = deep_learning(n_action = self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.mode_save_srv = rospy.Service('/model_save', Trigger, self.callback_model_save)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path)
        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.path_pose = PoseArray()
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.learning = True
        self.select_dl = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_'+str(self.mode)+'/'
        self.save_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/model_'+str(self.mode)+'/'
        self.load_path = roslib.packages.get_pkg_dir('nav_cloning') +'/data/analysis/model_gpu.pt'
        self.previous_reset_time = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.is_started = False
        self.start_time_s = rospy.get_time()
        os.makedirs(self.path + self.start_time)

        # with open(self.path + self.start_time + '/' +  'training.csv', 'w') as f:
        #     writer = csv.writer(f, lineterminator='\n')
        #     writer.writerow(['step', 'mode', 'loss', 'angle_error(rad)', 'distance(m)','x(m)','y(m)', 'the(rad)', 'direction'])
        self.tracker_sub = rospy.Subscriber("/tracker", Odometry, self.callback_tracker)

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_camera(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_camera(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_tracker(self, data):
        self.pos_x = data.pose.pose.position.x
        self.pos_y = data.pose.pose.position.y
        rot = data.pose.pose.orientation
        angle = tf.transformations.euler_from_quaternion((rot.x, rot.y, rot.z, rot.w))
        self.pos_the = angle[2]

    def callback_path(self, data):
        self.path_pose = data

    def callback_pose(self, data):
        distance_list = []
        pos = data.pose.pose.position
        for pose in self.path_pose.poses:
            path = pose.pose.position
            distance = np.sqrt(abs((pos.x - path.x)**2 + (pos.y - path.y)**2))
            distance_list.append(distance)

        if distance_list:
            self.min_distance = min(distance_list)


    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    def callback_dl_training(self, data):
        resp = SetBoolResponse()
        self.learning = data.data
        resp.message = "Training: " + str(self.learning)
        resp.success = True
        return resp

    def callback_model_save(self, data):
        model_res = SetBoolResponse()
        self.dl.save(self.save_path)
        model_res.message ="model_save"
        model_res.success = True
        return model_res

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return
        if self.vel.linear.x != 0:
            self.is_started = True
        if self.is_started == False:
            return
        img = resize(self.cv_image, (48, 64), mode='constant')
        mr_image = self.dl.MoRAM(img)
        conv1_img = self.dl.conv1_visualizing(img)
        conv2_img = self.dl.conv2_visualizing(conv1_img)
        conv3_img = self.dl.conv3_visualizing(conv2_img)
        # conv1_img = self.dl.feature_to_img(conv1_img)
        # conv2_img = self.dl.feature_to_img(conv2_img)
        conv3_img = self.dl.feature_to_img(conv3_img)
        
        # conv1Img0 = np.hstack((conv1_img[0], conv1_img[1], conv1_img[2], conv1_img[3]))
        # conv1Img1 = np.hstack((conv1_img[4], conv1_img[5], conv1_img[6], conv1_img[7]))
        # conv1Img2 = np.hstack((conv1_img[8], conv1_img[9], conv1_img[10], conv1_img[11]))
        # conv1Img3 = np.hstack((conv1_img[12], conv1_img[13], conv1_img[14], conv1_img[15]))
        # conv1Img = np.vstack((conv1Img0, conv1Img1, conv1Img2, conv1Img3))
    
        # conv2Img0 = np.hstack((conv2_img[0], conv2_img[1], conv2_img[2], conv2_img[3]))
        # conv2Img1 = np.hstack((conv2_img[4], conv2_img[5], conv2_img[6], conv2_img[7]))
        # conv2Img2 = np.hstack((conv2_img[8], conv2_img[9], conv2_img[10], conv2_img[11]))
        # conv2Img3 = np.hstack((conv2_img[12], conv2_img[13], conv2_img[14], conv2_img[15]))
        # conv2Img = np.vstack((conv2Img0, conv2Img1, conv2Img2, conv2Img3))
        
        conv3Img0 = np.hstack((conv3_img[0], conv3_img[1], conv3_img[2], conv3_img[3]))
        conv3Img1 = np.hstack((conv3_img[4], conv3_img[5], conv3_img[6], conv3_img[7]))
        conv3Img2 = np.hstack((conv3_img[8], conv3_img[9], conv3_img[10], conv3_img[11]))
        conv3Img3 = np.hstack((conv3_img[12], conv3_img[13], conv3_img[14], conv3_img[15]))
        conv3Img = np.vstack((conv3Img0, conv3Img1, conv3Img2, conv3Img3))
        
        # print(conv1_img.shape)
        
        # r, g, b = cv2.split(img)
        # img = np.asanyarray([r,g,b])

        img_left = resize(self.cv_left_image, (48, 64), mode='constant')
        #r, g, b = cv2.split(img_left)
        #img_left = np.asanyarray([r,g,b])

        img_right = resize(self.cv_right_image, (48, 64), mode='constant')
        #r, g, b = cv2.split(img_right)
        #img_right = np.asanyarray([r,g,b])
        ros_time = str(rospy.Time.now())

        if self.episode == 8000:
            self.learning = False
            self.dl.save(self.save_path)
            #self.dl.load(self.load_path)

        if self.episode == 10000:
            os.system('killall roslaunch')
            sys.exit()

        if self.learning:
            target_action = self.action
            distance = self.min_distance

            if self.mode == "manual":
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = 0
                action, loss = self.dl.act_and_trains(img , target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                angle_error = abs(action - target_action)

            elif self.mode == "zigzag":
                action, loss = self.dl.act_and_trains(img , target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                angle_error = abs(action - target_action)
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = 0

            elif self.mode == "use_dl_output":
                action, loss = self.dl.act_and_trains(img , target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                angle_error = abs(action - target_action)
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = action




            elif self.mode == "change_dataset_balance":
                if distance < 0.05:
                    action, loss = self.dl.act_and_trains(img , target_action)
                    if abs(target_action) < 0.1:
                        action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                elif 0.05 <= distance < 0.1:
                    self.dl.make_dataset(img , target_action)
                    action, loss = self.dl.act_and_trains(img , target_action)
                    if abs(target_action) < 0.1:
                        self.dl.make_dataset(img_left , target_action - 0.2)
                        action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                        self.dl.make_dataset(img_right , target_action + 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                    line = [str(self.episode), "training", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
                    with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(line)
                else:
                    self.dl.make_dataset(img , target_action)
                    self.dl.make_dataset(img , target_action)
                    action, loss = self.dl.act_and_trains(img , target_action)
                    if abs(target_action) < 0.1:
                        self.dl.make_dataset(img_left , target_action - 0.2)
                        self.dl.make_dataset(img_left , target_action - 0.2)
                        action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                        self.dl.make_dataset(img_right , target_action + 0.2)
                        self.dl.make_dataset(img_right , target_action + 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                    line = [str(self.episode), "training", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
                    with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(line)
                    with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(line)


                angle_error = abs(action - target_action)
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = action

            elif self.mode == "follow_line":
                action, loss = self.dl.act_and_trains(img , target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                angle_error = abs(action - target_action)

            elif self.mode == "selected_training":
                action = self.dl.act(img )
                angle_error = abs(action - target_action)
                loss = 0
                if angle_error > 0.05:
                    action, loss = self.dl.act_and_trains(img , target_action)
                    if abs(target_action) < 0.1:
                        action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                
                if distance > 0.15 or angle_error > 0.3:
                    self.select_dl = False
                # if distance > 0.1:
                #     self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = action

            # end mode

            self.episode += 1
            print(str(self.episode) + ", training, loss: " + str(loss) + ", angle_error: " + str(angle_error) + ", distance: " + str(distance))
            # print(str(self.episode)  + ", distance: " + str(distance))
            # line = [str(self.episode), "training", str(loss), str(angle_error), str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            line = [str(self.episode), "training", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        else:
            target_action = self.dl.act(img)
            distance = self.min_distance
            print(str(self.episode) + ", test, angular:" + str(target_action) + ", distance: " + str(distance))

            self.episode += 1
            angle_error = abs(self.action - target_action)
            # line = [str(self.episode), "test", "0", str(angle_error), str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            line = [str(self.episode), "test", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        temp = copy.deepcopy(img)
        cv2.imshow("Resized Image", temp)
        temp = copy.deepcopy(img_left)
        cv2.imshow("Resized Left Image", temp)
        temp = copy.deepcopy(img_right)
        cv2.imshow("Resized Right Image", temp)
        temp = copy.deepcopy(mr_image)
        temp = resize(temp, (480, 640))
        cv2.imshow("MoRAM Image", temp)
        # temp = copy.deepcopy(conv1Img)
        # temp = resize(temp, (440, 600))
        # cv2.imshow("Conv1 Image", temp)
        # temp = copy.deepcopy(conv2Img)
        # temp = resize(temp, (400, 560))
        # cv2.imshow("Conv2 Image", temp)
        # temp = copy.deepcopy(conv3Img)
        # temp = resize(temp, (480, 800))
        # cv2.imshow("Conv3 Image", temp)
        cv2.waitKey(1)

if __name__ == '__main__':
    rg = nav_cloning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
