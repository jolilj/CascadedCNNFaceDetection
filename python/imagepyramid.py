import numpy as np
from PIL import Image
import cv2
import imutils


## Pyramid representation of an image
#  labels are stored as [[cx,cy,w], [cx,cy,w]....]

class ImagePyramid:
    # Init with image
    def __init__(self, image, label,  scale=1.5,  minSize=(32,32)):
        self.constructPyramid(image, label, scale, minSize)
        self.scale = scale

    # Construct the image pyramid
    def constructPyramid(self, image, label,  scale, minSize):
        self.pyramid = []
        im = LabeledImage(image, label)
        self.pyramid.append(im)
        prevLabel = label
        while True:
            w = int(im.image.shape[1] / scale)
            label = np.round(prevLabel / scale)

            i = imutils.resize(im.image, width=w)
            im = LabeledImage(i, label)

            self.pyramid.append(im)
            prevLabel = label
            if (im.image.shape[0] < minSize[1] or im.image.shape[1] < minSize[0]):
                break
    
    # Returns the label expressed as top left corner and bottom right corner for display in matplotlib
    # [[topx, topy], [bottomx, bottomy]]
    def labelToRect(self):
        rects = []
        for i in self.pyramid:
            rects.append(i.labelToRect())
        return rects

class LabeledImage:
    def __init__(self,image, label):
        self.image = image
        self.label = label

    
def labelToRect(label):
    return [(int(label[0]-label[2]/2),int(label[1]-label[2]/2)),(int(label[0]+label[2]/2), int(label[1]+label[2]/2))]
