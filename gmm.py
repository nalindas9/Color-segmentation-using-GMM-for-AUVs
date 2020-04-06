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
import imutils
from imutils import contours

# seed random number generator
seed(1)

print('Headers Loaded!')
fig1, axs = plt.subplots(2,2)
# Function that plots BGR channels histogram for given image
def hist(image): 
  fig1.suptitle('BGR channel histograms for image')
  blue = cv2.calcHist([image],[0],None,[256],[0,256])
  blue = blue/blue.sum()
  green = cv2.calcHist([image],[1],None,[256],[0,256])
  green = green/green.sum()
  red = cv2.calcHist([image],[2],None,[256],[0,256])  
  red = red/red.sum()
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
      #  of point to lie in Gaussian 1
      p1 = (self.pi[0]*pdf(pixel, self.mu1, self.sigma1))/(self.pi[0]*pdf(pixel, self.mu1, self.sigma1) + self.pi[1]*pdf(pixel, self.mu2, self.sigma2) + self.pi[2]*pdf(pixel, self.mu3, self.sigma3))
      # Probability of point to lie in Gaussian 2
      p2 = (self.pi[1]*pdf(pixel, self.mu2, self.sigma2))/(self.pi[0]*pdf(pixel, self.mu1, self.sigma1) + self.pi[1]*pdf(pixel, self.mu2, self.sigma2) + self.pi[2]*pdf(pixel, self.mu3, self.sigma3))
      # Probability of point to lie in Gaussian 3
      p3 = (self.pi[2]*pdf(pixel, self.mu3, self.sigma3))/(self.pi[0]*pdf(pixel, self.mu1, self.sigma1) + self.pi[1]*pdf(pixel, self.mu2, self.sigma2) + self.pi[2]*pdf(pixel, self.mu3, self.sigma3))
      # list [pixel, probabilty to lie in gaussian1, probabilty to lie in gaussian2, probabilty to lie in gaussian3]
      probs.append([p1, p2, p3])
      self.probs = np.array(probs)
    return self.probs 
  
  # Function to find probability of each point in obtained gaussians
  def likelihood(self, pixel, cluster):
    if cluster == 'p1':
      # Probability of point to lie in Gaussian 1
      p1 = (self.pi[0]*pdf(pixel, self.mu1, self.sigma1))
      return p1
    elif cluster == 'p2':
      # Probability of point to lie in Gaussian 2
      p2 = (self.pi[1]*pdf(pixel, self.mu2, self.sigma2))
      return p2
    else:
      # Probability of point to lie in Gaussian 3
      p3 = (self.pi[2]*pdf(pixel, self.mu3, self.sigma3))
      return p3
      
      
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
   
    if self.log_likelihood[0]- log_likelihood[0] == 0 or self.log_likelihood[1]-log_likelihood[1] == 0 or self.log_likelihood[2]-log_likelihood[2] == 0:
      return True 
    else:
      # Updating the mean, std's, pi and log likelihood
      self.mu1, self.mu2,self.mu3, self.sigma1, self.sigma2, self.sigma3, self.pi, self.log_likelihood = new_mu[0],  new_mu[1],  new_mu[2], new_sigma[0], new_sigma[1], new_sigma[2], pi,log_likelihood 
      return False
         
  def train(self, iterations):
    for i in range(iterations):
      print('Iteration:', i+1)
      for count, pic in enumerate(sorted(glob.glob("/home/nalindas9/Documents/Courses/Spring_2020_Semester_2/ENPM673_Perception_for_Autonomous_Robots/Github/project3/Color-segmentation-using-GMM/DATANEW/Green_bouy"+ "/*")), 1):
        print('Image:', pic.split("Green_bouy/",1)[1])
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
    
    axs[0, 1].plot(xaxis, np.array(y1), linewidth=2, color='r')
    axs[0, 1].plot(xaxis, np.array(y2), linewidth=2, color='r')
    axs[0, 1].plot(xaxis, np.array(y3), linewidth=2, color='r')  
    axs[1, 1].plot(xaxis, y1, linewidth=2, color='r')
    axs[1, 1].plot(xaxis, y2, linewidth=2, color='r')
    axs[1, 1].plot(xaxis, y3, linewidth=2, color='r')
    
    plt.show()
  
  def test(self):
    cap = cv2.VideoCapture("/home/nalindas9/Documents/Courses/Spring_2020_Semester_2/ENPM673_Perception_for_Autonomous_Robots/Github/project3/Color-segmentation-using-GMM/DATANEW/detectbuoy.avi")
    if cap.isOpened() == False:
      print("Error loading video!")
    
    
    # Specify the path of the output video to be rendered
    out = cv2.VideoWriter('Green_buoy.avi',cv2.VideoWriter_fourcc(*'XVID'), 5, (640,480))

    #Iterating through all the frames in the Video
    print('Video Rendering started ...')
    no = 0
    while cap.isOpened():
      ret,frame = cap.read()
      if ret == False:
           break
      no = no+1  
      print('Frame no:', no)   
      green = np.array(frame[:,:,1])
      chance = np.vectorize(self.likelihood)
      pixel_probs1 = chance(green,'p1')
      pixel_probs2 = chance(green,'p2')
      pixel_probs3 = chance(green,'p3')
      pixel_probs1 = pixel_probs1.reshape(pixel_probs1.shape[0]*pixel_probs1.shape[1], 1)
      pixel_probs2 = pixel_probs2.reshape(pixel_probs1.shape[0]*pixel_probs1.shape[1], 1)
      pixel_probs3 = pixel_probs3.reshape(pixel_probs1.shape[0]*pixel_probs1.shape[1], 1)
      
      pixel_probs = np.concatenate((pixel_probs1, pixel_probs2, pixel_probs3), axis = 1)
      sum_pixel_probs = np.sum(pixel_probs, axis = 1)
      new_prob_image = sum_pixel_probs.reshape(green.shape[0], green.shape[1])
      new_prob_image[new_prob_image>(np.max(new_prob_image)/1.0001)] = 255 
      new_prob_image[new_prob_image<=(np.max(new_prob_image)/1.0001)] = 0
      #print('Pixel Probs:', pixel_probs2)
      #print('New image threshold:', new_prob_image)
      #print('Sum Pixel Probs:', sum_pixel_probs)
      #print('Image:',green)
      #edges = cv2.Canny(np.uint8(new_prob_image),20,255 )
      new_prob_image = cv2.GaussianBlur(new_prob_image,(5,5),0)
      kernel = np.ones((5,5), np.uint8) 
      new_prob_image = cv2.dilate(new_prob_image, kernel, iterations=1) 
      #cv2.imshow('Binary Image', new_prob_image)
      cnts,h = cv2.findContours(np.uint8(new_prob_image), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      if no <= 30:
      #img = cv2.drawContours(frame, cnts_sorted, 2, (0,255,0), 3)
        (cnts_sorted, boundingBoxes) = contours.sort_contours(cnts, method="left-to-right") 
        (x,y),radius = cv2.minEnclosingCircle(cnts_sorted[2])
        center = (int(x),int(y)+4)
        radius = int(radius)
        cv2.circle(frame,center,radius,(0,255,0),5)
        #print('Counts Sorted', cnts_sorted)
        out.write(frame)
      else:
        (cnts_sorted, boundingBoxes) = contours.sort_contours(cnts, method="right-to-left") 
        (x,y),radius = cv2.minEnclosingCircle(cnts_sorted[0])
        center = (int(x),int(y)+4)
        radius = int(radius)
        cv2.circle(frame,center,radius,(0,255,0),5)
        out.write(frame)
        
      #cv2.imshow('Contour Image', frame)
      cv2.waitKey(0)
      
      
      
    out.release()
    cap.release()
    cv2.destroyAllWindows()  
      
  
    
def main():
  em = GmmEm(3)
  em.train(30)
  em.test()
  



if __name__ == '__main__':
  main()




