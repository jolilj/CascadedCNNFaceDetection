import numpy as np
from PIL import Image
import cv2
import imutils

class ImagePyramid:
    # Init with image
    def __init__(self, image, scale=1.5,  minSize=(32,32)):
        self.pyramid = self.constructPyramid(image,scale, minSize)
        self.scale = scale
    # Construct the image pyramid
    def constructPyramid(self, image, scale, minSize):
        self.pyramid = []
        self.pyramid.append(image)
        im = image
        while True:
            w = int(im.shape[1] / scale)
            im = imutils.resize(image, width=w)
            print(im.shape)
            self.pyramid.append(im)
            if (im.shape[0] < minSize[1] or im.shape[1] < minSize[0]):
                break

            

