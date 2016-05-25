import random
import glob
import numpy as np
import cv2
import math
from imagepyramid import ImagePyramid
from PIL import Image
import matplotlib.pyplot as plt
import imageloader as il
import slidingwindow as sw
import numpy as np
import cv2
import imutils

INITIAL_DOWNSCALE = 2

# Preprocessing functions for images before giving them as input to the 24-net


## Load and process a single image from a given path
def loadAndPreProcessSingle(path, pyramidScale, pyramidMinSize):
    image = loadAndNormalize(path + 'jpg')
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
                tempIm = loadAndNormalize('images/' + prevLine + '.jpg')
                getNextLine = 1
            elif(getNextLine):
                label = getLabel(line)
                imdb.append(ImagePyramid(tempIm,label, pyramidScale, pyramidMinSize))
                getNextLine = 0
            prevLine = line
    return imdb

def loadAndPreProcessNegative(path, N,  pyramidScale, pyramidMinSize):
    im_paths = glob.glob(path + '/*.jpg')
    rand_idx = random.sample(range(0, len(im_paths)), N)
    imdb = []
    for i in rand_idx:
        # Put an annotation window outside of the image to ensure a label=0
        label = np.asarray([-10000, -10000, 100])
        tempIm = loadAndNormalize(im_paths[i])
        imdb.append(ImagePyramid(tempIm,label, pyramidScale, pyramidMinSize))
    return imdb
# Generate subimages from the sliding window approach and return in format compatible with the CNN.
def getCNNFormat(imdb, stepSize, windowSize):
    X = []
    Y = []
    W = []
    for k in range(0,len(imdb)):
        [windows, labels, croppedImages] = sw.slideWindow(imdb[k], k, stepSize, windowSize)
        for i in range(0,len(croppedImages)):
            for j in range(0,len(croppedImages[i])):
                X.append(croppedImages[i][j])
                Y.append(labels[i][j])
                W.append(np.asarray(windows[i][j]))
    X = np.asarray(X)
    X = X.reshape(X.shape[0],1,X.shape[1], X.shape[2])
    Y = np.asarray(Y)
    Y=Y.reshape(Y.shape[0],1)
    W = np.asarray(W)
    W = W.reshape(W.shape[0],1,W.shape[1])
    return [X,Y,W]

# Generate CNN format without sliding window given a set of subimages from an original image
# references (coordinates etc for windows) are stored in windowPos
def getCNNFormatSingle(imdb, windowSize, windowPos):
    X = []
    Y = []
    W = []
    for k in range(0,len(imdb)):
        #print("processing image, no sliding: {}".format(k+1))
        imagePyramid = imdb[k]
        for i in range(0,len(imagePyramid.pyramid)):
            label = imagePyramid.pyramid[i].label
            image = imagePyramid.pyramid[i].image
            # Get center of the image
            x_c = int(image.shape[1]/2)
            y_c = int(image.shape[0]/2)
            if (windowSize <= min(image.shape[0], image.shape[1])):
                y = y_c-windowSize/2
                x = x_c-windowSize/2
                subImage = image[y_c-windowSize/2:y_c+windowSize/2,x_c-windowSize/2:x_c+windowSize/2]
                scaleFactor = math.pow(imagePyramid.scale,i)
                #Set window info related to original image (shift back to original image coordinate system)
                windowInfo = [windowPos[k][0] + int(x_c*scaleFactor), windowPos[k][1] +  int(y_c*scaleFactor),int(windowSize*scaleFactor),windowPos[k][2]]
                #Calculate new label
                #Get image label if available
                if (type(label) != int):
                    labelwidth = label[2]
                    xlabel_left = label[0]-int(labelwidth/2)
                    xlabel_right = label[0]+int(labelwidth/2)
                    ylabel_upper = label[1]-int(labelwidth/2)
                    ylabel_lower = label[1]+int(labelwidth/2)
                    
                    #Compare to window and calculate new label
                    margin = 2/math.pow(labelwidth,2)
                    sublabelx = 1- margin*(math.pow(x-xlabel_left,2)+ math.pow(x+windowSize-xlabel_right,2))
                    sublabelx = max(sublabelx, 0.0)
                    sublabely = 1- margin*(math.pow(y-ylabel_upper,2)+ math.pow(y+windowSize-ylabel_lower,2))
                    sublabely = max(sublabely, 0.0)
                    sublabel = min(sublabelx, sublabely)
                else:
                    sublabel = label
                X.append(subImage)
                Y.append(sublabel)
                W.append(np.asarray(windowInfo))

    X = np.asarray(X)
    X = X.reshape(X.shape[0],1,X.shape[1], X.shape[2])
    Y = np.asarray(Y)
    Y=Y.reshape(Y.shape[0],1)
    W = np.asarray(W)
    W = W.reshape(W.shape[0],1,W.shape[1])
    return [X,Y,W]
# load an image from file and perform basic pre-processing
def loadAndNormalize(path):
    pil_im = Image.open(path).convert('L')
    img = np.array(pil_im)
    img = img - np.mean(img)
    img = img/np.std(img)
    img = imutils.resize(img, int(math.ceil(img.shape[1]/INITIAL_DOWNSCALE)))
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
	return np.asarray([x_center, y_center, math.fabs(width)])/INITIAL_DOWNSCALE
