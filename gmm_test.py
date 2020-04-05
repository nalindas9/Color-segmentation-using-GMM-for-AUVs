"""
Expectation test algorithm

Authors:
-Nalin Das (nalindas9@gmail.com), 
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""

def main():
  cap = cv2.VideoCapture("/home/nalindas9/Documents/Courses/Spring_2020_Semester_2/ENPM673_Perception_for_Autonomous_Robots/Github/project3/Color-segmentation-using-GMM/DATANEW/detectbuoy.avi")
  if cap.isOpened() == False:
    print("Error loading video!")
  
  while cap.isOpened():
    ret,frame = cap.read()
    print("Frame:",Frame)
    if ret == False:
         break
    img = cv2.resize(frame, (0,0),fx=0.5,fy=0.5)
    cv2.imshow('Image', img)

if __name__ == '__main__':
  main()
