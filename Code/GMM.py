# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob


'''
Author: Aditya Khopkar, Graduate Robotics Student, UMD
Coursework: ENPM673

'''

class GMM:
    def __init__(self,data,n_clusters,parameters,mode="1D"):
        self.data = data
        self.n = data.shape[0]
        self.K = n_clusters
        self.mode = mode
        self.num = np.zeros(self.K)
        self.den = np.zeros(self.K)
        self.log_likelihoods = [0]
        self.init_parameters = parameters
        self.new_mean = np.zeros_like(parameters['mean'])
        self.new_cov = np.zeros_like(parameters['cov'])
        self.new_w = np.zeros_like(parameters['w'])

    def calcGaussianProbability(self,mean,covariance):
        data = self.data
        mode = self.mode
        if mode == "1D":
            std = np.sqrt(covariance)
            diff = (data-mean)
            gauss_prob = 1/(np.sqrt(2*np.pi)*std) * np.exp(-0.5*((diff)/(std))**2)
        if mode == "3D":
            cov_det = np.linalg.det(covariance)
            cov_inv = np.linalg.det(covariance)
            diff = np.matrix(data-mean)
            gauss_prob = (2.0*np.pi)**(-data.shape[1]/2.0) * (1.0 / (cov_det**0.5)) * np.exp(-0.5 * np.sum(np.multiply(diff*cov_inv,diff) , axis=1))
        gauss_prob[gauss_prob == 0] = 0.0000192
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
        data = self.data
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
            #gauss_prob = mvn.pdf(self.data,m[k],cov[k],allow_singular=True)
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
        #thresh = self.thresh
        log_likelihoods = self.log_likelihoods
        for i in range(0,iterations):
            responsibility,log_likelihood = self.E_step(parameters)
            log_likelihoods.append(log_likelihood)
            parameters = self.M_step(responsibility)
            loss = np.abs(log_likelihoods[-1] - log_likelihoods[-2])
            print('Loss for epoch '+str(i)+' : '+str(loss)+' likelihood '+str(log_likelihood))
        
        return parameters




