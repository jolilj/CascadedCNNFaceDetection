import cv2
import imutils
import matplotlib.pyplot as plt
from imagepyramid import ImagePyramid

image = cv2.imread('test_im.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
pyr  = ImagePyramid(gray)



