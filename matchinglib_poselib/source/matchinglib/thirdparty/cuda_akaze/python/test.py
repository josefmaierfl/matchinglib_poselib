import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2

import libakaze_pybindings as akaze
import numpy as np

options = akaze.AKAZEOptions()

img1 = cv2.imread("../../datasets/iguazu/img1.pgm",0)
img2 = cv2.imread("../../datasets/iguazu/img2.pgm",0)

img1_32 = np.float32(img1);
img1_32 = img1_32*(1./255.)

img2_32 = np.float32(img2);
img2_32 = img2_32*(1./255.)

height, width = np.shape(img1)
options.setWidth(width)
options.setHeight(height)
evolution1 = akaze.AKAZE(options)

height, width = np.shape(img2)
options.setWidth(width)
options.setHeight(height)
evolution2 = akaze.AKAZE(options)

evolution1.Create_Nonlinear_Scale_Space(img1_32)
desc1,kpts1 = evolution1.Compute_Descriptors()

evolution2.Create_Nonlinear_Scale_Space(img2_32)
desc2,kpts2 = evolution2.Compute_Descriptors()

matcher = akaze.Matcher()
dmatch = matcher.BFMatch(desc1,desc2)

