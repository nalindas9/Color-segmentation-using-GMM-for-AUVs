import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import dataGenerator as dg
from GMM import GMM

#Train Yellow
def train_yellow():
	print('###################################################################')
	print("Training Yellow ...")
	data1 = dg.generate_data_yellow()
	#print('Data size: ',data1.shape)
	#global K 
	K1 = 2
	p1 = dg.initialize_parameters(data1,K1)
	#print('Initial Parameters:', p1)
	gmm = GMM(data1,K1,p1)
	parameters1 = gmm.solve(50)
	#print('Parameters:'+str(parameters1))
	print("Training Done.")
	return parameters1

#Train Orange
def train_orange():
	print('###################################################################')
	print("Training Orange ...")
	data2 = dg.generate_data_orange()
	#print('Data size: ',data2.shape)
	#global K 
	K2 = 2
	p2 = dg.initialize_parameters(data2,K2)
	#print('Initial Parameters:', p)
	gmm = GMM(data2,K2,p2)
	parameters2 = gmm.solve(50)
	#print('Parameters:'+str(parameters2))
	print("Training Done.")
	return parameters2

#Train Green
def train_green():
	print('###################################################################')
	print("Training Green ...")
	data3 = dg.generate_data_green()
	#print('Data size: ',data3.shape)
	#global K 
	K3 = 3
	p3 = dg.initialize_parameters(data3,K3)
	#print('Initial Parameters:', p)
	gmm = GMM(data3,K3,p3)
	parameters3 = gmm.solve(50)
	#print('Parameters:'+str(parameters3))
	print("Training Done.")
	return parameters3