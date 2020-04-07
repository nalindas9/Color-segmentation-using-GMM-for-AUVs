import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob




"""
Initialize parameters Randomly : mean, std and weight
"""

def initialize_parameters(data,K):
    np.random.seed(1)
    n,d = data.shape
    mean = data[np.random.choice(n,K,replace=False)]
    covariance = [80*np.eye(d)] * K
    weights = [1./K] * K
    for i in range(K):
        covariance[i] = np.multiply(covariance[i],np.random.rand(d,d))

    parameters = {
        'mean':mean,
        'cov':covariance,
        'w':weights
    }
    return parameters

def generate_data_yellow(mode = "1D"):
    data = []
    for fname in glob.glob('/home/akhopkar/DATANEW/Yellow_buoy/*'):
        img = cv2.imread(fname)
        resized = cv2.resize(img,(40,40),interpolation=cv2.INTER_LINEAR)
        img = resized[15:25,15:25]
        if mode == "1D":
            img_red_channel = img[:,:,2]
            img_green_channel = img[:,:,1]
            nx,ny,_ = img.shape
            redChannel = np.reshape(img_red_channel,((nx*ny),1))
            greenChannel = np.reshape(img_green_channel,((nx*ny),1))
            #yellow2D = np.vstack((redChannel,greenChannel)).T
            yello = np.concatenate((redChannel,greenChannel),axis=0)
        if mode == "3D":
            nx,ny,ch = img.shape
            yello = np.reshape(img,((nx*ny),ch))
        for i in range(yello.shape[0]):
            data.append(yello[i,:])
    data = np.array(data)
    return data

def generate_data_orange():
    data = []
    for fname in glob.glob('/home/akhopkar/DATANEW/Orange_buoy/*'):
        img = cv2.imread(fname)
        resized = cv2.resize(img,(40,40),interpolation=cv2.INTER_LINEAR)
        img = resized[15:25,15:25]
        img_red_channel = img[:,:,2]
        nx,ny,_ = img.shape
        red1D = np.reshape(img_red_channel,((nx*ny),1))
        for i in range(red1D.shape[0]):
            data.append(red1D[i,:])
    data = np.array(data)
    return data

def generate_data_green():
    data = []
    for fname in glob.glob('/home/akhopkar/DATANEW/Green_buoy/*'):
        img = cv2.imread(fname)
        resized = cv2.resize(img,(40,40),interpolation=cv2.INTER_LINEAR)
        img = resized[15:25,15:25]
        img_green_channel = img[:,:,1]
        #img_blue_channel = img[:,:,0]
        nx,ny,_ = img.shape
        #greenC = np.reshape(img_green_channel,((nx*ny),1))
        #blueC = np.reshape(img_blue_channel,((nx*ny),1))
        #green1D = np.concatenate((blueC,greenC),axis=0)
        
        green1D = np.reshape(img_green_channel,((nx*ny),1))
        #yellow2D = np.vstack((redChannel,greenChannel)).T
        for i in range(green1D.shape[0]):
            data.append(green1D[i,:])
    data = np.array(data)
    return data
