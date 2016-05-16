import cv2
import imutils
import matplotlib.pyplot as plt
from imagepyramid import ImagePyramid
import numpy as np

image = cv2.imread('test_im.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
label = np.asarray([int(gray.shape[1]/2), int(gray.shape[0]/2), 150])
pyr  = ImagePyramid(gray,label)

for i in pyr.pyramid:
    rect = i.labelToRect()
    print(rect[0])
    print(rect[1])
    cv2.rectangle(i.image, rect[0] , rect[1], [0,255,0], 1)
    plt.imshow(i.image, cmap = 'gray', interpolation = 'bicubic')
    plt.show()

