import numpy as np
import cv2
import math
from imagepyramid import ImagePyramid
from PIL import Image
#import matplotlib.pyplot as plt


def loadAndPreProcessIms(annotationsFile, pyramidScale, pyramidMinSize):
    paths = []
    ellipses = []
    imdb = []
    tempIm = 0 #Init temp image variable
    with open(annotationsFile) as inputfile:
        prevLine = 0
        getNextLine = 0
        for line in inputfile:
            line = line.strip()
            if(len(line)==1 and int(line)==1):
                #Load image, normalize and make zero mean
                tempIm = loadAndNormalize(prevLine)
                getNextLine = 1
            elif(getNextLine):
                label = getLabel(line)
                imdb.append(ImagePyramid(tempIm,label, pyramidScale, pyramidMinSize))
                getNextLine = 0
            prevLine = line
    return imdb

def loadAndNormalize(path):
    pil_im = Image.open('images/' + path + '.jpg').convert('L')
    img = np.array(pil_im)
    img = img - np.mean(img)
    img = img/np.std(img)
    return img

def getLabel(line):
	ellipse = line.split(' ')
	rmax = float(ellipse[0])
	rmin = float(ellipse[1])
	angle = float(ellipse[2])
	x_center = float(ellipse[3])
	y_center = float(ellipse[4])
	width = int(2*rmax*math.cos(angle*math.pi/180))
	#ellipsedata.append([(int(x_center), int(y_center)), [int(angle)], (int(rmax), int(rmin))])
	return np.asarray([x_center, y_center, math.fabs(width)])


#imdb = loadAndPreProcessIms('annotations.txt', 1.5, (32,32))
#pyr = imdb[0]

#rect = pyr.pyramid[1].labelToRect()
#cv2.rectangle(pyr.pyramid[1].image, rect[0], rect[1], [0, 255, 0],1 )

#plt.imshow(pyr.pyramid[0].image,cmap=plt.cm.gray)
#plt.show()

#print("===================")
#plt.imshow(pyr.pyramid[1].image,cmap=plt.cm.gray)
#plt.show()
