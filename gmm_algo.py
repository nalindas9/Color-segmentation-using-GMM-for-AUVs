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
        #self.obs = data.shape[1]
        self.K = n_clusters
        self.thresh = thresh
        self.num = np.zeros(self.K)
        self.den = np.zeros(self.K)
        self.log_likelihoods = []
        self.init_parameters = parameters
        self.new_mean = np.zeros_like(self.init_parameters['mean'])
        self.new_cov  = np.zeros_like(self.init_parameters['cov'])
        self.new_w = np.zeros_like(self.init_parameters['w'])

    def calcGaussianProbability(self,mean,covariance):
        data = self.data
        gauss_prob = 1/(math.sqrt(2*np.pi)*covariance) * np.exp(-0.5*((data-mean)/(covariance))**2)
        return gauss_prob
    
    def getParams(self,params):
        mean = params['mean']
        cov = params['cov']
        w = params['w']
        return mean,cov,w
        
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
        alpha = (z/np.sum(z,axis=0)).T
        log_likelihood = np.sum(np.log(np.sum(z, axis = 0)))
        #log_likelihoods.append(log_likelihood)
        return alpha, log_likelihood
    
    def M_step(self,alpha):
        data = self.data
        #w = self.w
        n = self.n
        K = self.K
        new_mean = self.new_mean
        new_cov = self.new_cov
        new_w = self.new_w
        num = self.num
        den = self.den
        for k in range(K):
#             mean[k] = 1. / N_ks[k] * np.sum(z[:, k] * data.T, axis = 1).T
#             x_mean = np.matrix(data - mean[k])
#             cov[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mean.T,  z[:, k]), x_mean))
#             w[k] = 1. / n * N_ks[k]
            num[k] = np.dot(alpha[k,:],data)
            den[k] = np.sum(alpha[k,:])
            new_mean[k] = num[k]/den[k]
            x_mean = np.matrix(data-new_mean[k])
            new_cov[k] = 1./den[k] * np.dot(np.multiply(x_mean.T,alpha[k,:]),x_mean) 
            new_w[k] = 1/n * den[k]
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
            print('Epoch:'+str(i))
            #self.mean,self.cov,self.w = self.getParams(parameters)
            responsibility,log_likelihood = self.E_step(parameters)
            print('Likelihood:',log_likelihood)
            log_likelihoods.append(log_likelihood)
            parameters = self.M_step(responsibility)
            if len(log_likelihoods) < 2: 
                continue
            if len(log_likelihoods) > 1000 or np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < thresh:
                break
        
        return parameters


'''
Initialize parameters based on no. of clusters and the data shape
return: Initial parameters - Dictionary
'''
def initialize_parameters(data,K):
    n,d = data.shape
    mean = yello1D[np.random.choice(n,K,replace=False)]
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

