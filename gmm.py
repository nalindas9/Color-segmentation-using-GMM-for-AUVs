"""
Expectation Maximization Algorithm

Authors:
-Nalin Das (nalindas9@gmail.com), 
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from random import seed
from random import random
from scipy.stats import norm
# seed random number generator
seed(1)

print('Headers Loaded!')
fig1, axs = plt.subplots(2,2)
# Function that plots BGR channels histogram for given image
def hist(image): 
  fig1.suptitle('BGR channel histograms for image')
  blue = cv2.calcHist([image],[0],None,[256],[0,256])
  green = cv2.calcHist([image],[1],None,[256],[0,256])
  red = cv2.calcHist([image],[2],None,[256],[0,256])  
  axs[0, 0].plot(blue,color = 'b')
  axs[0, 1].plot(green,color = 'g')
  axs[1, 0].plot(red,color = 'r')
  return blue, green, red

# K is the number of Gaussians that can be fit into the histogram
# For the image Green_boi_42.png, K is 3

# Function to calculate pdf of a given gaussian
def pdf(x, mu, sigma):
  #print('x:', x, 'mu:', mu, 'sigma:', sigma, ((1/(sigma*np.sqrt(2*np.pi))*np.exp((-(x-mu)**2)/(2*sigma**2)))))
  return ((1/(sigma*np.sqrt(2*np.pi))*np.exp((-(x-mu)**2)/(2*sigma**2))))

"""
# Function to generate univariate gaussian to fit given gaussian
def gaussian(img, data):
  mu = np.mean(data)
  sigma = np.std(data)
  print(data)
  print('mean', data.mean())
  x = np.linspace(0, 255, 256)
  axs[1,1].plot(x, pdf(x, mu, sigma), linewidth=2, color='r')
"""

# GMM and EM estimation class
class GmmEm:
  # Initializing with the three gaussians which would like to mix
  def __init__(self, channel, k):
    self.channel = channel.flatten()
    self.k = k
    self.mix = 1/self.k
    self.pi =[random()]*self.k
    self.mu1, self.sigma1, self.mu2, self.sigma2, self.mu3, self.sigma3 = 180,15,240,15, 255, 3
    # Variable to store the probalities of the datapoint lying in all three gaussians
    self.probs = []
    
    # Function that generates initial random gaussians for the image
  def generate_gauss(self):
    mu1, mu2, mu3, sigma1, sigma2, sigma3 =  self.mu1, self.mu2,self.mu3, self.sigma1, self.sigma2, self.sigma3
    plt.title('Fitted Gaussian {:}: μ = {:}, σ = {:}, {:}: μ = {:}, σ = {:}, {:}: μ = {:}, σ = {:}'.format("1", mu1, sigma1, "2", mu2, sigma2, "3", mu3, sigma3))
    gauss1 = np.random.normal(mu1, sigma1, 100)
    gauss2 = np.random.normal(mu2, sigma2, 100)
    gauss3 = np.random.normal(mu3, sigma3, 100)
    count1, bins1, ignored1 = plt.hist(gauss1, 30, density=True, color = 'b')
    count2, bins2, ignored2 = plt.hist(gauss2, 30, density=True, color = 'b')
    count3, bins3, ignored3 = plt.hist(gauss3, 30, density=True, color = 'b')
    plt.plot(bins1, 1/(sigma1 * np.sqrt(2 * np.pi))*np.exp( - (bins1 - mu1)**2 / (2 * sigma1**2) ), linewidth=2, color='r')
    plt.plot(bins2, 1/(sigma2 * np.sqrt(2 * np.pi))*np.exp( - (bins2 - mu2)**2 / (2 * sigma2**2) ), linewidth=2, color='r')
    plt.plot(bins3, 1/(sigma3 * np.sqrt(2 * np.pi))*np.exp( - (bins3 - mu3)**2 / (2 * sigma3**2) ), linewidth=2, color='r')
    plt.show()
    data = np.concatenate((gauss1, gauss2, gauss3), axis=0)
    print('Fitted Gaussian {:}: μ = {:}, σ = {:}'.format("1", mu1, sigma1))
    print('Fitted Gaussian {:}: μ = {:}, σ = {:}'.format("2", mu2, sigma2))
    print('Fitted Gaussian {:}: μ = {:}, σ = {:}'.format("3", mu3, sigma3))
  
  
  # Perform Estimation step
  def e_step(self):
    probs = []
    # Finding probability of each point 
    for pixel in self.channel:
      # Probability of point to lie in Gaussian 1
      p1 = (self.pi[0]*pdf(pixel, self.mu1, self.sigma1))/(self.pi[0]*pdf(pixel, self.mu1, self.sigma1) + self.pi[1]*pdf(pixel, self.mu2, self.sigma2) + self.pi[2]*pdf(pixel, self.mu3, self.sigma3))
      # Probability of point to lie in Gaussian 2
      p2 = (self.pi[1]*pdf(pixel, self.mu2, self.sigma2))/(self.pi[0]*pdf(pixel, self.mu1, self.sigma1) + self.pi[1]*pdf(pixel, self.mu2, self.sigma2) + self.pi[2]*pdf(pixel, self.mu3, self.sigma3))
      # Probability of point to lie in Gaussian 3
      p3 = (self.pi[2]*pdf(pixel, self.mu3, self.sigma3))/(self.pi[0]*pdf(pixel, self.mu1, self.sigma1) + self.pi[1]*pdf(pixel, self.mu2, self.sigma2) + self.pi[2]*pdf(pixel, self.mu3, self.sigma3))
      # list [pixel, probabilty to lie in gaussian1, probabilty to lie in gaussian2, probabilty to lie in gaussian3]
      probs.append([p1, p2, p3])
      self.probs = np.array(probs)
    return self.probs 
  
  # Perform Maximization step
  def m_step(self):
    sum_clusters = []
    # Calculate sum of point probs for each cluster 1,2 and 3
    for cluster in range(len(self.probs[0])):
      summ = np.sum(self.probs[:, cluster])
      sum_clusters.append(summ)
    
    pi = []
    new_mu = []
    new_sigma = []
    for clusterr,sumc in enumerate(sum_clusters,0):
      # Calculate fraction of point's belonging to each cluster 1,2 and 3
      pi.append(sumc/np.sum(sum_clusters)) 
      # Calculate new updated mean
      mu = (np.sum(((self.channel).reshape(len(self.channel),1))*(self.probs[:,clusterr].reshape(len(self.probs[:,clusterr]),1))))/sumc
      new_mu.append(mu)
      # Calculate new updated standard deviation      
      new_sigma.append(((np.sum((self.probs[:,clusterr].reshape(len(self.probs[:,clusterr]),1))*(((self.channel).reshape(len(self.channel),1))-mu)**2))/sumc)**(1/2))
    
    # Updating the mean and std's
    self.mu1, self.mu2,self.mu3, self.sigma1, self.sigma2, self.sigma3 = new_mu[0],  new_mu[1],  new_mu[2], new_sigma[0], new_sigma[1], new_sigma[2]
        
    return new_sigma
      

     
def main():
  img = cv2.imread('/home/nalindas9/Documents/Courses/Spring_2020_Semester_2/ENPM673_Perception_for_Autonomous_Robots/Github/project3/Color-segmentation-using-GMM/Green_boi_42.png')
  k = 3
  blue, green, red = hist(img)
  img = img[:,:,1]
  #gaussian(img, green)
  em = GmmEm(img, k)
  em.generate_gauss()
  probs = em.e_step()
  #print('The probabilties is:', probs)
  probs1 = em.m_step()
  print ('New variance is:', probs1)
  em.generate_gauss()
  #print(np.sum(probs,axis=1))
  #print('Data:', data)
  #print('Reshaped gausses:', (img.flatten()).reshape(len(img.flatten()),1)*(probs[:,0].reshape(len(probs),1)))
  #plt.hist(data)
  plt.show()



if __name__ == '__main__':
  main()




