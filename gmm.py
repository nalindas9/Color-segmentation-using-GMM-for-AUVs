"""
Expectation Maximization Algorithm

Authors:
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Plotting RGB channel histograms and observing the value of K from this
img = cv2.imread('/home/nalindas9/Documents/Courses/Spring_2020_Semester_2/ENPM673_Perception_for_Autonomous_Robots/Github/project3/Color-segmentation-using-GMM/Green_boi_42.png')
color = ('b','g','r')
for i,col in enumerate(color):
  histr = cv2.calcHist([img],[i],None,[256],[0,256])
  plt.plot(histr,color = col)
  plt.xlim([0,256])
  plt.show()
  
# K is the number of Gaussians that can be fit into the histogram
# For the image Green_boi_42.png, K is 5



