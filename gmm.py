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
  def __init__(self, channel, gauss1, gauss2, gauss3, k):
    self.gauss1 = gauss1
    self.gauss2 = gauss2
    self.gauss3 = gauss3
    self.channel = channel.flatten()
    self.k = k
    self.mix = 1/self.k
    self.pi =[random()]*self.k
    self.mu1, self.sigma1, self.mu2, self.sigma2, self.mu3, self.sigma3 = np.mean(gauss1), np.std(gauss1), np.mean(gauss2), np.std(gauss2), np.mean(gauss3), np.std(gauss3)
    # Variable to store the probalities of the datapoint lying in all three gaussians
    self.probs = []
  # Perform Estimation step
  def e_step(self):
    # Finding probability of each point 
    for pixel in self.channel:
      p1 = (self.pi[0]*pdf(pixel, self.mu1, self.sigma1))/(self.pi[0]*pdf(pixel, self.mu1, self.sigma1) + self.pi[1]*pdf(pixel, self.mu2, self.sigma2) + self.pi[2]*pdf(pixel, self.mu3, self.sigma3))
      p2 = (self.pi[1]*pdf(pixel, self.mu2, self.sigma2))/(self.pi[0]*pdf(pixel, self.mu1, self.sigma1) + self.pi[1]*pdf(pixel, self.mu2, self.sigma2) + self.pi[2]*pdf(pixel, self.mu3, self.sigma3))
      p3 = (self.pi[2]*pdf(pixel, self.mu3, self.sigma3))/(self.pi[0]*pdf(pixel, self.mu1, self.sigma1) + self.pi[1]*pdf(pixel, self.mu2, self.sigma2) + self.pi[2]*pdf(pixel, self.mu3, self.sigma3))
      # list [pixel, probabilty to lie in gaussian1, probabilty to lie in gaussian2, probabilty to lie in gaussian3]
      self.probs.append([pixel, p1, p2, p3])
    return self.probs
      
# Function that generates initial random gaussians for the image
def generate_gauss():
  mu1, mu2, mu3, sigma1, sigma2, sigma3 = 180, 240, 255, 15 , 15, 3
  gauss1 = np.random.normal(mu1, sigma1, 100)
  gauss2 = np.random.normal(mu2, sigma2, 100)
  gauss3 = np.random.normal(mu3, sigma3, 100)
  count1, bins1, ignored1 = plt.hist(gauss1, 30, density=True, color = 'b')
  count2, bins2, ignored2 = plt.hist(gauss2, 30, density=True, color = 'b')
  count3, bins3, ignored3 = plt.hist(gauss3, 30, density=True, color = 'b')
  axs[1,1].plot(bins1, 1/(sigma1 * np.sqrt(2 * np.pi))*np.exp( - (bins1 - mu1)**2 / (2 * sigma1**2) ), linewidth=2, color='r')
  axs[1,1].plot(bins2, 1/(sigma2 * np.sqrt(2 * np.pi))*np.exp( - (bins2 - mu2)**2 / (2 * sigma2**2) ), linewidth=2, color='r')
  axs[1,1].plot(bins3, 1/(sigma3 * np.sqrt(2 * np.pi))*np.exp( - (bins3 - mu3)**2 / (2 * sigma3**2) ), linewidth=2, color='r')
  plt.show()
  data = np.concatenate((gauss1, gauss2, gauss3), axis=0)
  print('Input Gaussian {:}: μ = {:}, σ = {:}'.format("1", mu1, sigma1))
  print('Input Gaussian {:}: μ = {:}, σ = {:}'.format("2", mu2, sigma2))
  print('Input Gaussian {:}: μ = {:}, σ = {:}'.format("3", mu3, sigma3))
  return data, gauss1, gauss2, gauss3



     
def main():
  img = cv2.imread('/home/nalindas9/Documents/Courses/Spring_2020_Semester_2/ENPM673_Perception_for_Autonomous_Robots/Github/project3/Color-segmentation-using-GMM/Green_boi_42.png')
  k = 3
  blue, green, red = hist(img)
  img = img[:,:,1]
  #gaussian(img, green)
  data, gauss1, gauss2, gauss3 = generate_gauss()
  em = GmmEm(img, gauss1, gauss2, gauss3, k)
  probs = em.e_step()
  print('The probabilties is:', probs)
  #plt.hist(data)
  plt.show()



if __name__ == '__main__':
  main()




