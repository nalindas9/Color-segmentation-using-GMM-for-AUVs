import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
from scipy.stats import multivariate_normal as mvn
import dataGenerator as dg
from GMM import GMM
import trainer as tr


parameters1 = tr.train_yellow()
parameters2 = tr.train_orange()
parameters3 = tr.train_green()
#print(parameters3)

def prep_img(test_img,color):
  n,d,_ = test_img.shape
  if color == "Yellow":
    print("Prepping Yellow..")
    img_red_channel = test_img[:,:,2]
    img_green_channel = test_img[:,:,1]
    redChannel = np.reshape(img_red_channel,((n*d),1))
    greenChannel = np.reshape(img_green_channel,((n*d),1))
    #yellow2D = np.vstack((redChannel,greenChannel)).T
    img = np.concatenate((redChannel,greenChannel),axis=0)
    data = []
    for i in range(img.shape[0]):
        data.append(img[i,:])
    
  if color == "Green":
    #print(n*d)
    print("Prepping Green..")
    img_G = test_img[:,:,1]
    #img_B = test_img[:,:,0]
    #blueChannel = np.reshape(img_B,((n*d),1))
    greenChannel = np.reshape(img_G,((n*d),1))
    #img = np.concatenate((greenChannel,blueChannel),axis=0)
    img = greenChannel
    data = []
    for i in range(img.shape[0]):
        data.append(img[i,:])

  if color == "Orange":
    print("Prepping Orange..")
    img = test_img[:,:,2]
    ch = 1
    img = np.reshape(img, (n*d,ch))
    data = []
    for i in range(img.shape[0]):
        data.append(img[i,:])

  return np.array(data)

def mvn_pdf(data,mean,cov,allow_singular=True):
	if allow_singular == True:
		std = np.sqrt(cov)
		diff = (data-mean)
		gauss_prob = 1/(np.sqrt(2*np.pi)*std) * np.exp(-0.5*((diff)/(std))**2)
		gauss_prob[gauss_prob==0] = 0.000192
	return gauss_prob

w1 = parameters1['w']
mu1 = parameters1['mean']
sigma1 = parameters1['cov']

w2 = parameters2['w']
mu2 = parameters2['mean']
sigma2 = parameters2['cov']

w3 = parameters3['w']
mu3 = parameters3['mean']
sigma3 = parameters3['cov']

cap = cv2.VideoCapture('/home/akhopkar/detectbuoy.avi')
Frame = 0
images = []
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Buoy_detection.avi',fourcc,5.0,(640,480))

while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == False:
        break
    test_img = frame
    #cv2.imshow('buoy',frame)
    K1 = 2   #For orange,yellow
    K2 = 2
    K3 = 3  #For green
    n,d,_ = test_img.shape
    print(n,d)
    #print(n*d)

    #Generate Test Data for each buoy
    img1 = prep_img(test_img,"Yellow")
    img2 = prep_img(test_img,"Orange")
    img3 = prep_img(test_img,"Green")
    print(img1.shape)
    
    # Create object for each test image - Yellow Orange Green
    test1 = GMM(img1,K1,parameters1)
    test2 = GMM(img2,K1,parameters2)
    test3 = GMM(img2,K1,parameters3)

    #For each buoy define the shape of probability
    prob1 = np.zeros((K1,img1.shape[0]))
    z1 = np.zeros((img1.shape[0],K1))
    posterior_prob1 = np.zeros((img1.shape[0],K1))
    print(prob1.shape)

    prob2 = np.zeros((K2,img2.shape[0]))
    z2 = np.zeros((img2.shape[0],K2))
    posterior_prob2 = np.zeros((img2.shape[0],K2))
    print(prob2.shape)

    prob3 = np.zeros((K3,img3.shape[0]))
    z3 = np.zeros((img3.shape[0],K3))
    posterior_prob3 = np.zeros((img3.shape[0],K3))
    print(prob3.shape)

    ## PROBABILITY for YELLOW and ORANGE
    for k in range(K1):
        # #prob = w[k]*test.calcGaussianProbability(mu[k],sigma[k])
        prob1[k] = w1[k]*mvn.pdf(img1,mu1[k],sigma1[k],allow_singular = True)
        prob2[k] = w2[k]*mvn.pdf(img2,mu2[k],sigma2[k],allow_singular = True)
        # prob1 = w1[k]*test1.calcGaussianProbability(mu1[k],sigma1[k])
        # prob2 = w2[k]*test2.calcGaussianProbability(mu2[k],sigma2[k])
        #prob1 = w1[k]*pdf(img1,mu1[k],sigma1[k])
        #prob2 = w2[k]*pdf(img2,mu2[k],sigma2[k])
        
        #z[:,k] = prob.reshape((n,))
        posterior_prob1 = prob1.sum(0)
        posterior_prob2 = prob2.sum(0)
#     print('Posterior',posterior_prob)    
    
    ##PROBABILITY For GREEN
    for k in range(K3):
        prob3[k] = w3[k]*mvn.pdf(img3,mu3[k],sigma3[k],allow_singular = True)
        #prob3 = w3[k]*test3.calcGaussianProbability(mu3[k],sigma3[k])
        posterior_prob3 = prob3.sum(0)
    
    #Probability arrangement for yellow
    temp_g = posterior_prob1[:n*d]
    temp_r = posterior_prob1[n*d:]
    red = temp_r
    green = temp_g
    prob1=np.add(red,green)
    prob1 = prob1/2

    prob1[prob1 > np.max(prob1) / 1.5] = 255
    prob1[prob1 < np.max(prob1) / 1.5] = 0
    prob1 = np.reshape(prob1,(n,d))

    output1 = np.zeros_like(frame[:,:,0])
    output1[:,:] = prob1

    #cv2_imshow(output1)

    #Probability arrangement for orange
    prob2 = np.reshape(posterior_prob2,(n,d))
    prob2[prob2>np.max(prob2)/3.0] = 255
    
    output2 = np.zeros_like(frame[:,:,2])
    output2[:,:] = prob2
    
    #Probability arrangement for Green
    prob3 = np.reshape(posterior_prob3,(n,d))
    prob3[prob3 > np.max(prob3) / 4] = 255
    prob3[prob3 < np.max(prob3) / 4] = 0
    #prob = np.reshape(prob,(n,d))
    output3 = np.zeros_like(frame[:,:,0])
    output3[:,:] = prob3
    
    #Morphological for Yellow  
    kernel1 = np.ones((2,2),dtype = np.uint8)
    dilate_yellow = cv2.dilate(output1,kernel1, iterations = 18)
    erode_yellow = cv2.erode(dilate_yellow,kernel1,iterations=20)
    open_yellow = cv2.dilate(erode_yellow,kernel1,iterations = 50)
    contour_yellow,_ = cv2.findContours(open_yellow,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    #Morphological for Orange
    kernel2 = np.ones((2,2),dtype=np.uint8)
    erode_orange = cv2.erode(output2,kernel2,iterations=5)
    dilate_orange = cv2.dilate(erode_orange,kernel2, iterations = 18)
    contour_orange,_ = cv2.findContours(dilate_orange,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    #Morphological for Green
    blur_green = cv2.GaussianBlur(output3,(7,7),5)
    _,edge_green = cv2.threshold(blur_green,110,255,0)
    kernel3 = np.ones((2,2), np.uint8) 
    dilate_green = cv2.dilate(output3,kernel3,iterations=15)
    contour_green,_ = cv2.findContours(np.uint8(dilate_green), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #print("#####################################################################")
    
    ### CONTOUR PROCESS FOR ALL THE BUOYS ###
    print("FRame",Frame)
    for contour in contour_yellow:
        if 4000<cv2.contourArea(contour) <10500:
            n_contours = len(contour_yellow)
            if n_contours > 1:
                max_area = max([cv2.contourArea(contour) for contour in contour_yellow if cv2.contourArea(contour)<10500 ])
                
                if cv2.contourArea(contour)< max_area:
                    continue
           # print("area",cv2.contourArea(contour))
            (x1,y1),radius1 = cv2.minEnclosingCircle(contour)
            if 6<radius1:
                cv2.circle(frame,(int(x1)-43,int(y1)-40),int(radius1)-30,(0,180,255),2)

    for contour in contour_orange:
        #print("Area",cv2.contourArea(contour))
        if 4.0 < cv2.contourArea(contour):
            #print("Area within",cv2.contourArea(contour))
            (x2,y2),radius2 = cv2.minEnclosingCircle(contour)
            #print("Radius",radius2)
            if 4<radius2:
                cv2.circle(test_img,(int(x2)-10,int(y2)-15),int(radius2)-3,(0,0,255),2)
    
    for contour in contour_green:
        area3 = cv2.contourArea(contour)
        # print("Area of contours",area)
        if 1250<area3<1800:
            #print("Bounded area",area3)
            #(cnts_sorted, boundingBoxes) = contours.sort_contours(contour, method = "Left-to-right")
            (x3,y3),radius3 = cv2.minEnclosingCircle(contour)
            if 20 < radius3 <27: 
                cv2.circle(test_img,(int(x3)-5, int(y3)-5), 14,(0,255,0),2)

    cv2.imshow('Final',test_img)
    out.write(test_img)
    Frame+=1
    if cv2.waitKey(5) & 0xFF == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
