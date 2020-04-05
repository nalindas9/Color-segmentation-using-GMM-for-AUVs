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
import glob

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
   new_pdf = ((1/(sigma*np.sqrt(2*np.pi))*np.exp((-(x-mu)**2)/(2*(sigma**2)))))
   #print ('Pdf is:', new_pdf)
   return new_pdf

  

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
  def __init__(self, k):
    self.k = 3
    self.pi =[random()]*self.k
    self.log_likelihood = [random()]*self.k
    self.mu1, self.sigma1, self.mu2, self.sigma2, self.mu3, self.sigma3 = 180,15,240,15, 255, 3
    self.gauss1 = np.random.normal(self.mu1, self.sigma1, 100)
    self.gauss2 = np.random.normal(self.mu2, self.sigma2, 100)
    self.gauss3 = np.random.normal(self.mu3, self.sigma3, 100)
    # Variable to store the probalities of the datapoint lying in all three gaussians
    self.probs = []
    
    # Function that generates initial random gaussians for the image
  def generate_gauss(self):
    mu1, mu2, mu3, sigma1, sigma2, sigma3 =  self.mu1, self.mu2,self.mu3, self.sigma1, self.sigma2, self.sigma3
    plt.title('Fitted Gaussian {:}: μ = {:}, σ = {:}, {:}: μ = {:}, σ = {:}, {:}: μ = {:}, σ = {:}'.format("1", mu1, sigma1, "2", mu2, sigma2, "3", mu3, sigma3))
    
    data = np.concatenate((self.gauss1, self.gauss2, self.gauss3), axis=0)
    #print('Fitted Gaussian {:}: μ = {:}, σ = {:}'.format("1", mu1, sigma1))
    #print('Fitted Gaussian {:}: μ = {:}, σ = {:}'.format("2", mu2, sigma2))
    #print('Fitted Gaussian {:}: μ = {:}, σ = {:}'.format("3", mu3, sigma3))
  
  
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
    log_likelihood = []
    for clusterr,sumc in enumerate(sum_clusters,0):
      # Calculate fraction of point's belonging to each cluster 1,2 and 3
      pi.append(sumc/np.sum(sum_clusters)) 
      # Calculate new updated mean
      mu = (np.sum(((self.channel).reshape(len(self.channel),1))*(self.probs[:,clusterr].reshape(len(self.probs[:,clusterr]),1))))/sumc
      new_mu.append(mu)
      # Calculate new updated standard deviation      
      new_sigma.append(((np.sum((self.probs[:,clusterr].reshape(len(self.probs[:,clusterr]),1))*(((self.channel).reshape(len(self.channel),1))-mu)**2))/sumc)**(1/2))
      log_likelihood.append(np.sum(np.log(sumc)))
    
    print('new sigma:', new_sigma)
    if self.log_likelihood[0]- log_likelihood[0] == 0 or self.log_likelihood[1]-log_likelihood[1] == 0 or self.log_likelihood[2]-log_likelihood[2] == 0:
      return True 
    else:
      # Updating the mean, std's, pi and log likelihood
      self.mu1, self.mu2,self.mu3, self.sigma1, self.sigma2, self.sigma3, self.pi, self.log_likelihood = new_mu[0],  new_mu[1],  new_mu[2], new_sigma[0], new_sigma[1], new_sigma[2], pi,log_likelihood 
      return False
         
  def train(self, iterations):
    for i in range(iterations):
      print('Iteration:', i+1)
      for count, pic in enumerate(sorted(glob.glob("/home/nalindas9/Documents/Courses/Spring_2020_Semester_2/ENPM673_Perception_for_Autonomous_Robots/Github/project3/Color-segmentation-using-GMM/DATANEW/Green_bouy/training_data"+ "/*")), 1):
        print('Image:', pic.split("training_data/",1)[1])
        img = cv2.imread(pic) 
        k = 3
        blue, green, red = hist(img)
        img = img[:,:,1]
        self.channel = img.flatten()
        self.generate_gauss()
        probs = self.e_step()
        converge = self.m_step()
        if converge == True:
          return
        print('')
      print('Fitted Gaussian {:}: μ = {:}, σ = {:}, weight = {:}, log likelihood = {:}'.format("1", self.mu1, self.sigma1, self.pi[0], self.log_likelihood[0]))
      print('Fitted Gaussian {:}: μ = {:}, σ = {:}, weight = {:}'.format("2", self.mu2, self.sigma2, self.pi[1], self.log_likelihood[1]))
      print('Fitted Gaussian {:}: μ = {:}, σ = {:}, weight = {:}'.format("3", self.mu3, self.sigma3, self.pi[2], self.log_likelihood[2]))
      print('')
      
    y1, y2, y3 = [], [], []
    xaxis = np.sort(self.channel)
    for point in xaxis:
      y1.append(1/(self.sigma1 * np.sqrt(2 * np.pi))*np.exp( - (point - self.mu1)**2 / (2 * self.sigma1**2) ))
      y2.append(1/(self.sigma2 * np.sqrt(2 * np.pi))*np.exp( - (point - self.mu2)**2 / (2 * self.sigma2**2) ))
      y3.append(1/(self.sigma3 * np.sqrt(2 * np.pi))*np.exp( - (point - self.mu3)**2 / (2 * self.sigma3**2) ))
    
    axs[0, 1].plot(xaxis, np.array(y1)*(10/max(y2)), linewidth=2, color='r')
    axs[0, 1].plot(xaxis, np.array(y2)*(10/max(y2)), linewidth=2, color='r')
    axs[0, 1].plot(xaxis, np.array(y3)*(10/max(y2)), linewidth=2, color='r')  
    axs[1, 1].plot(xaxis, y1, linewidth=2, color='r')
    axs[1, 1].plot(xaxis, y2, linewidth=2, color='r')
    axs[1, 1].plot(xaxis, y3, linewidth=2, color='r')
    
    plt.show()
    
    return self.mu1, self.mu2,self.mu3, self.sigma1, self.sigma2, self.sigma3, self.pi
  
    
def main():
  em = GmmEm(3)
  mu1, mu2, mu3, sigma1, sigma2, sigma3, weights = em.train(20)
  
  file1 = open("predicted_gmm.txt", "w")
  file1.write(str(mu1))
  file1.write("\n")
  file1.write(str(mu2))
  file1.write("\n")
  file1.write(str(mu3))
  file1.write("\n")
  file1.write(str(sigma1))
  file1.write("\n")
  file1.write(str(sigma2))
  file1.write("\n")
  file1.write(str(sigma3))
  file1.write("\n")
  file1.write(str(weights))
  file1.write("\n")

  file1.close()



if __name__ == '__main__':
  main()




