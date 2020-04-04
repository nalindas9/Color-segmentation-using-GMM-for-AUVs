import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import glob

'''
Author: Aditya Khopkar, Graduate Robotics Student @ UMD
Coursework: ENPM673

'''

class gmm_algo:
    def __init__(self,data,n_clusters,parameters,thresh):
        self.data = data
        self.n = data.shape[0]
        self.K = n_clusters
        self.thresh = thresh
        self.num = np.zeros(K)
        self.den = np.zeros(K)
        self.log_likelihoods = [0]
        self.init_parameters = parameters
        self.new_mean = np.zeros_like(parameters['mean'])
        self.new_cov = np.zeros_like(parameters['cov'])
        self.new_w = np.zeros_like(parameters['w'])

    def calcGaussianProbability(self,mean,covariance):
        data = self.data
        gauss_prob = 1/(np.sqrt(2*np.pi)*covariance) * np.exp(-0.5*((data-mean)/(covariance))**2)
        return gauss_prob
    
    def getParams(self,params):
        mean = params['mean']
        cov = params['cov']
        w = params['w']
        return mean,cov,w
        
    def update_covariance(self,alpha,mean):
        den = np.sum(alpha,axis=0)
        cov = self.new_cov
        K = self.K
        data = self.data
        for k in range(0,K):
            x_mean = data-mean[k,:]
            alpha_diag = np.diag(alpha[:,k])
            x = np.matrix(x_mean)
            sigma = x.T * alpha_diag * x
            cov[k,:,:] = sigma/den[k]
        return cov

    def update_mean(self,alpha):
        data = self.data
        den = np.sum(alpha,axis=0)
        mean = self.new_mean
        mean = np.dot(alpha.T,data)/den[:,np.newaxis]
        return mean

    def update_weight(self,alpha):
        n,d = data.shape
        w = self.new_w
        w = np.mean(alpha,axis = 0)
        return w
        
    def E_step(self,params):
        m = params['mean']
        cov = params['cov']
        weights = params['w']
        n = self.n
        K = self.K
        z = np.zeros((self.n,K))
        for k in range(K):
            gauss_prob = self.calcGaussianProbability(m[k],cov[k])
            temp = weights[k]*gauss_prob
            z[:,k] = temp.reshape((n,))
        alpha = (z.T/np.sum(z,axis=1)).T
        log_likelihood = np.sum(np.log(np.sum(z, axis = 1)))
        return alpha, log_likelihood
    
    def M_step(self,alpha):
        new_mean = self.update_mean(alpha)
        new_cov = self.update_covariance(alpha,new_mean)
        new_w = self.update_weight(alpha)
        updated_parameters = {
            'mean':new_mean,
            'cov':new_cov,
            'w':new_w
        }
        return updated_parameters
    
    def solve(self,iterations):
        parameters = self.init_parameters
        thresh = self.thresh
        log_likelihoods = self.log_likelihoods
        for i in range(0,iterations):
            responsibility,log_likelihood = self.E_step(parameters)
            log_likelihoods.append(log_likelihood)
            parameters = self.M_step(responsibility)
            loss = np.abs(log_likelihoods[-1] - log_likelihoods[-2])
            print('Loss for epoch '+str(i)+' : '+str(loss)+' likelihood '+str(log_likelihood))
        
        return parameters


'''
Initialize parameters based on no. of clusters and the data shape
return: Initial parameters - Dictionary
'''
def initialize_parameters(data,K):
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

'''
Generate data : stacking all the training data set
'''
def generate_data_yellow():
	data = []
	for fname in glob.glob('/home/akhopkar/Desktop/PROJECT_3/yellow_train/*'):
	    img = cv2.imread(fname)
	    resized = cv2.resize(img,(40,40),interpolation=cv2.INTER_LINEAR)
	    img = resized[15:25,15:25]
	#         cv2.imshow('file',img)
	#     # #     if cv2.waitKey(5) & 0xFF == 27:
	#     # #         break
	#         cv2.waitKey(0)
	    img_red_channel = img[:,:,2]
	    img_green_channel = img[:,:,1]
	    nx,ny,_ = img.shape
	    redChannel = np.reshape(img_red_channel,((nx*ny),1))
	    greenChannel = np.reshape(img_green_channel,((nx*ny),1))
	    #yellow2D = np.vstack((redChannel,greenChannel)).T
	    yello1D = np.concatenate((redChannel,greenChannel),axis=0)
	    for i in range(yello1D.shape[0]):
	        data.append(yello1D[i,:])
	data = np.array(data)
	return data

def generate_data_orange():
	data = []
	for fname in glob.glob('/home/akhopkar/Desktop/PROJECT_3/orange_train/*'):
	    img = cv2.imread(fname)
	    resized = cv2.resize(img,(40,40),interpolation=cv2.INTER_LINEAR)
	    img = resized[15:25,15:25]
	#         cv2.imshow('file',img)
	#     # #     if cv2.waitKey(5) & 0xFF == 27:
	#     # #         break
	#         cv2.waitKey(0)
	    img_red_channel = img[:,:,2]
	    nx,ny,_ = img.shape
	    red1D = np.reshape(img_red_channel,((nx*ny),1))
	    for i in range(red1D.shape[0]):
	        data.append(red1D[i,:])
	data = np.array(data)
	return data

def generate_data_green():
	data = []
	for fname in glob.glob('/home/akhopkar/Desktop/PROJECT_3/green_train/*'):
	    img = cv2.imread(fname)
	    resized = cv2.resize(img,(40,40),interpolation=cv2.INTER_LINEAR)
	    img = resized[15:25,15:25]
	#         cv2.imshow('file',img)
	#     # #     if cv2.waitKey(5) & 0xFF == 27:
	#     # #         break
	#         cv2.waitKey(0)
	    img_green_channel = img[:,:,1]
	    nx,ny,_ = img.shape
	    green1D = np.reshape(img_green_channel,((nx*ny),1))
	    #yellow2D = np.vstack((redChannel,greenChannel)).T
	    for i in range(green1D.shape[0]):
	        data.append(green1D[i,:])
	data = np.array(data)
	return data

TRAIN_MODE = 'Yellow'



if TRAIN_MODE == 'Yellow':
	data = generate_data_yellow()
if TRAIN_MODE == 'Orange':
	data = generate_data_orange()
if TRAIN_MODE == 'Green':
	data = generate_data_green()
print('Data size: ',data.shape)
K = 2
p = initialize_parameters(data,K) 
print('Initial Parameters:', p)
gmm = gmm_algo(data,K,p,0.0001)
parameters = gmm.solve(50)
print('Parameters:'+str(parameters))
# hist_red = cv2.calcHist([img_red_channel],[0],None,[256],[0,256]) 
# hist_green = cv2.calcHist([img_green_channel],[0],None,[256],[0,256])
# hist_yellow = np.concatenate((hist_red,hist_green),axis = 0)
# hist_yellow2 = hist_red+hist_green

# plt.plot(hist_yellow2,color='black')
# plt.xlim([0,256])
# plt.show

