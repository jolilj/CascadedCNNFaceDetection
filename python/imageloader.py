import numpy as np
import cv2
import math
from imagepyramid import ImagePyramid
from PIL import Image
import imageloader as il
import slidingwindow as sw
import numpy as np
import cv2

## Load and process a single image from a given path
def loadAndPreProcessSingle(path, pyramidScale, pyramidMinSize):
    image = loadAndNormalize(path)
    return ImagePyramid(image, -1, pyramidScale, pyramidMinSize)

# Load and process images given an annotationsfile containing image path and label
# according to the dataset
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

# Generate subimages from the sliding window approach and return in format compatible with the CNN.
def getCNNFormat(imdb, stepSize, windowSize):
    X = []
    Y = []
    W = []
    for i in range(0,len(imdb)):
        print("processing image: {}".format(i+1))
        [windows, labels, croppedImages] = sw.slideWindow(imdb[i], i, stepSize, windowSize)
        for i in range(0,len(croppedImages)):
            for j in range(0,len(croppedImages[i])):
                X.append(croppedImages[i][j])
                Y.append(labels[i][j])
                W.append(np.asarray(windows[i][j]))
    X = np.asarray(X)
    print("inside il")
    print(X.shape)
    print("=====")
    X = X.reshape(X.shape[0],1,X.shape[1], X.shape[2])
    Y = np.asarray(Y)
    Y=Y.reshape(Y.shape[0],1)
    W = np.asarray(W)
    W = W.reshape(W.shape[0],1,W.shape[1])
    return [X,Y,W]

# load an image from file and perform basic pre-processing
def loadAndNormalize(path):
    pil_im = Image.open('images/' + path + '.jpg').convert('L')
    img = np.array(pil_im)
    img = img - np.mean(img)
    img = img/np.std(img)
    return img

# generate a square label from the given ellipse data
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
