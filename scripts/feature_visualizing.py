#!/usr/bin/env python3
from __future__ import print_function

import roslib
roslib.load_manifest('nav_cloning')
import numpy as np
import math
import matplotlib as plt
import os
import time
from os.path import expanduser

import roslib
roslib.load_manifest('nav_cloning')
import rospy
from sensor_msgs.msg import Image
# from nav_cloning_pytorch import *
from skimage.transform import resize
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from yaml import load
from PIL import Image
import cv2

# HYPER PARAM
BATCH_SIZE = 8
MAX_DATA = 10000


class Visualizing_Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    #<Network CNN 3 + FC 2> 
        self.conv1 = nn.Conv2d(n_channel, 32,kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64,64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(960, 512)
        self.fc5 = nn.Linear(512,n_out)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(1, 1, 8, stride=4)
        self.deconv2 = nn.ConvTranspose2d(1, 1, 3, stride=2)
        self.deconv3 = nn.ConvTranspose2d(1, 1, 3, stride=1)
        self.average1 = torch.zeros((1, 1, 11, 15))
        self.average2 = torch.zeros((1, 1, 5, 7))
        self.average3 = torch.zeros((1, 1, 3, 5))
        self.ave0 = 0
        self.ave1 = 0
        self.ave2 = 0
        self.ave3 = 0

    #<Weight set>
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        torch.nn.init.ones_(self.deconv1.weight)
        torch.nn.init.ones_(self.deconv2.weight)
        torch.nn.init.ones_(self.deconv3.weight)
        #self.maxpool = nn.MaxPool2d(2,2)
        #self.batch = nn.BatchNorm2d(0.2)
        self.flatten = nn.Flatten()

    #<FC layer (output)>
        self.fc_layer = nn.Sequential(
            self.fc4,
            self.relu,
            self.fc5,
        )

        # self.weight1 = torch.ones((1, 1, 8, 8))
        # self.weight2 = torch.ones((1, 1, 3, 3))
        # self.deconv1.weight.data = self.weight1
        # self.deconv2.weight.data = self.weight2
        # self.deconv3.weight.data = self.weight2

    
    #<forward layer>
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.relu(x1)
        in1 = x2.to('cpu').detach().numpy().copy()
        for i in in1:
            for j in i:
                self.ave1 += j
        self.ave1 /= 32
        # print('conv1')
        # print(self.ave1)
        ave1_img = Image.fromarray(self.ave1)
        ave1_img = ave1_img.resize((64, 48))
        self.ave1 = np.array(ave1_img)
        # ave1_img.show()
        # self.ave1 = torch.from_numpy(self.ave1.astype(np.float32)).clone()  #resize
        # self.ave1 = torch.from_numpy(ave1_array.astype(np.float32)).clone()
        # self.ave1 = self.ave1.to('cuda')
        x3 = self.conv2(x2)
        x4 = self.relu(x3)
        in2 = x4.to('cpu').detach().numpy().copy()
        for i in in2:
            for j in i:
                self.ave2 += j
        self.ave2 /= 64
        ave2_img = Image.fromarray(self.ave2)
        ave2_img = ave2_img.resize((64, 48))
        self.ave2 = np.array(ave2_img)
        # ave2_img.show()
        # self.ave2 = torch.from_numpy(self.ave2.astype(np.float32)).clone()
        # self.ave2 = torch.from_numpy(ave2_array.astype(np.float32)).clone()
        # self.ave2 = self.ave2.to('cuda')
        x5 = self.conv3(x4)
        self.feature = x5
        x6 = self.relu(x5)
        in3 = x6.to('cpu').detach().numpy().copy()
        for i in in3:
            for j in i:
                self.ave3 += j
        self.ave3 /= 64
        ave3_img = Image.fromarray(self.ave3)
        ave3_img = ave3_img.resize((64, 48))
        self.ave3 = np.array(ave3_img)
        # ave3_img.show()
        # self.ave3 = torch.from_numpy(self.ave3.astype(np.float32)).clone()
        # self.ave3 = torch.from_numpy(ave3_array.astype(np.float32)).clone()
        # self.ave3 = self.ave3.to('cuda')
        x7 = self.flatten(x6)
        x8 = self.fc_layer(x7)
        return x8

    def feature2image(self):
        # ave1_reshape = torch.reshape(self.ave1, (1, 1, 11, 15))
        # ave2_reshape = torch.reshape(self.ave2, (1, 1, 5, 7))
        # ave3_reshape = torch.reshape(self.ave3, (1, 1, 3, 5))
        # image = self.deconv3(ave3_reshape) * ave2_reshape
        # image = torch.sqrt(image)
        # image = self.deconv2(image) * ave1_reshape
        # image = torch.sqrt(image)
        # image = self.deconv1(image)
        # image = torch.reshape(image, (48, 64))
        # image = self.ave3 * self.ave2 
        # image = np.sqrt(image)
        # image = image * self.ave1
        # image = np.sqrt(image)
        # return image
        image = self.ave3 + self.ave2 
        # image = np.sqrt(image)
        image = image / 2
        image = image + self.ave1
        # image = np.sqrt(image)
        image = image / 2
        return image
    
    def feature(self):
        feature = self.feature
        return feature

class feature_visualizing:
    def __init__(self, n_channel=3, n_action=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Visualizing_Net(n_channel, n_action)
        self.net.to(self.device)
        print(self.device)
        self.optimizer = optim.Adam(self.net.parameters(),eps=1e-2,weight_decay=5e-4)
        self.totensor = transforms.ToTensor()
        # self.n_action = n_action
        self.transform=transforms.Compose([transforms.ToTensor()])
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/288.jpg'
        self.load_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/analysis/model_gpu.pt'
        self.net.load_state_dict(torch.load(self.load_path))
        self.img = Image.open(self.path)
        self.img_array = np.asarray(self.img)
        self.flag = True

    def test(self):
        self.net.eval()
        x_ten = torch.tensor(self.img_array,dtype=torch.float32, device=self.device).unsqueeze(0)
        x_ten = x_ten.permute(0,3,1,2)
        self.net(x_ten)
        feature_img = self.net.feature2image()
        # feature_img = feature_img.permute(0,2,3,1)
        # feature_img = feature_img.data.numpy()
        # feature_img = torch.reshape(feature_img, (48, 64))
        # feature_img = feature_img.to('cpu').detach().numpy().copy()
        # print(feature_img)
        pil_img = Image.fromarray(feature_img)
        # pil_img = pil_img.resize((640, 480))
        pil_img.show()

if __name__ == '__main__':
        fv = feature_visualizing()
        fv.test()
