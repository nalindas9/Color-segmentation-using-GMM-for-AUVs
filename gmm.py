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

# K is the number of Gaussians that can be fit into the histogram
# For the image Green_boi_42.png, K is 3

# Gaussian class
class Gaussian:
  def __init__(self, mu, sigma):
    self.mu = mu
    self.sigma = sigma
    
  def pdf(self, x):
    return ((1/(self.sigma*np.sqrt(2*np.pi))*np.exp((-(x-self.mu)**2)/(2*self.sigma**2))))


# Function that generates initial random gaussians for the image
def generate_gauss():
  mu1, mu2, mu3, sigma1, sigma2, sigma3 = 180, 240, 255, 15 , 15, 3
  gauss1 = np.random.normal(mu1, sigma1, 100)
  gauss2 = np.random.normal(mu2, sigma2, 100)
  gauss3 = np.random.normal(mu3, sigma3, 100)
  count1, bins1, ignored1 = plt.hist(gauss1, 30, density=True, color = 'b')
  count2, bins2, ignored2 = plt.hist(gauss2, 30, density=True, color = 'b')
  count3, bins3, ignored3 = plt.hist(gauss3, 30, density=True, color = 'b')
  axs[1,1].plot(bins1, 1/(sigma1 * np.sqrt(2 * np.pi))*np.exp( - (bins1 - mu1)**2 / (2 * sigma1**2) ), linewidth=2, color='y')
  axs[1,1].plot(bins2, 1/(sigma2 * np.sqrt(2 * np.pi))*np.exp( - (bins2 - mu2)**2 / (2 * sigma2**2) ), linewidth=2, color='y')
  axs[1,1].plot(bins3, 1/(sigma3 * np.sqrt(2 * np.pi))*np.exp( - (bins3 - mu3)**2 / (2 * sigma3**2) ), linewidth=2, color='y')
  plt.show()
  data = np.concatenate((gauss1, gauss2, gauss3), axis=0)
  print('Input Gaussian {:}: μ = {:}, σ = {:}'.format("1", mu1, sigma1))
  print('Input Gaussian {:}: μ = {:}, σ = {:}'.format("2", mu2, sigma2))
  print('Input Gaussian {:}: μ = {:}, σ = {:}'.format("3", mu3, sigma3))
  return data
  
# Expectation Maximization Algorithm Function
#def ex_max():
     
def main():
  img = cv2.imread('/home/nalindas9/Documents/Courses/Spring_2020_Semester_2/ENPM673_Perception_for_Autonomous_Robots/Github/project3/Color-segmentation-using-GMM/Green_boi_42.png')
  hist(img)
  data = generate_gauss()
  gauss_fit = Gaussian(np.mean(data), np.std(data))
  plt.hist(data)
  plt.show()



if __name__ == '__main__':
  main()




