# Color segmentation using GMM for AUVs

[![License:MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/nalindas9/Color-segmentation-using-GMM-for-AUVs/blob/master/LICENSE)
 
## Authors
Nalin Das, Aditya Khopkar, Nidhi Bhojak

## About
This is the repository for the project - Color segmentation using GMM for AUVs. In this project, the goal was to detect and segment buoys of different colors underwater using the Gaussian Mixture Model and the Expectation-Maximization algorithm. Result: >80% frames buoys accurately segmented.

<img src = "images/ezgif-2-9af811c4b14e.gif">

## System and Dependencies
- OpenCV2
- Numpy
- Matplotlib
- Scipy
- Ubuntu 16.04 LTS

## How to run
To execute the program:

1. Make sure the Training set and the code files are in same folder.
2. Make sure the input video 'detectbuoy.avi' is in the same folder as code files.
3. The output video will be written in the same folder as the code file

The Code folder contains the following executable files:
1. dataGenerator.py
2. GMM.py
3. trainer.py
4. detection.py

To execute the program, run the command $python3 detection.py$
