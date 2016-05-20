from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten
from keras.optimizers import SGD
from tempfile import TemporaryFile
import imageloader as il
from imagepyramid import ImagePyramid
import slidingwindow as sw
import numpy as np
import cv2
import time
import model_architecture
import os
import sys
import train48Net as train
import math

sliding_windowSize = 24
minSize = (sliding_windowSize*2, sliding_windowSize*2)
scaleFactor = 2
stepSize = 24
batchSize = 16
nbEpoch = 1
zoomFactor = 3

## Load data for processing and then send into first net
# If preprocessed files exists (data path passed as argument) load the raw data
if (len(sys.argv) > 1):
    print("======Loading data from file...======")
    start_time = time.time()
    data_path = str(sys.argv[1])
    X = np.load(data_path + '_X.npy')
    Y = np.load(data_path + '_Y.npy')
    W = np.load(data_path + '_W.npy')
    print("finished loading in {0}".format(time.time()-start_time))
# Otherwise load images and preprocess
else:
    print("======Loading and Preprocessing...======")
    start_time = time.time()
    imdb = il.loadAndPreProcessIms('annotations_train_short.txt', scaleFactor, (sliding_windowSize,sliding_windowSize))
    
    [X, Y, W] = il.getCNNFormat(imdb, stepSize, sliding_windowSize)
    np.save('data/data_X',X)
    np.save('data/data_Y',Y)
    np.save('data/data_W',W)
    print("finished preprocessing in {0}".format(time.time()-start_time))


print("X-shape: {0}".format(X.shape))
print("Y-shape: {0}".format(Y.shape))

idx = np.squeeze(Y>0.5)
valid_windows = W[idx,:,:]
imdb_48 = []
for i in range(0,valid_windows.shape[0]):
    window = np.squeeze(valid_windows[i,:,:])
    pyr = imdb[window[3]]
    image = pyr.image
    label = pyr.pyramid[0].label
    valid_windows[i,:,2] = window[2]*zoomFactor
    y1 = int(window[1]-sliding_windowSize*zoomFactor/2)
    y2 = int(window[1]+sliding_windowSize*zoomFactor/2)
    x1 = int(window[0]-sliding_windowSize*zoomFactor/2)
    x2 = int(window[0]+sliding_windowSize*zoomFactor/2)
    if (not (y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1])):  
        subImage = image[y1:y2,x1:x2]
        print(subImage.shape)
        windowSize = sliding_windowSize*zoomFactor
        labelwidth = label[2]
        xlabel_left = label[0]-int(labelwidth/2)
        xlabel_right = label[0]+int(labelwidth/2)
        ylabel_upper = label[1]-int(labelwidth/2)
        ylabel_lower = label[1]+int(labelwidth/2)

        #Compare to window and calculate new label
        margin = 1.5/math.pow(labelwidth,2)
        sublabelx = 1- margin*(math.pow(x1-xlabel_left,2)+ math.pow(x2-xlabel_right,2))
        sublabelx = max(sublabelx, 0)
        sublabely = 1- margin*(math.pow(y1-ylabel_upper,2)+ math.pow(y2-ylabel_lower,2))
        sublabely = max(sublabely, 0)
        sublabel = min(sublabelx, sublabely)

        imdb_48.append(ImagePyramid(subImage, sublabel, scaleFactor, (windowSize/4,windowSize/4) ))

print("===VALID IMS====")
X_48 = []
Y_48 = []
W_48 = valid_windows
for im in imdb_48:
    height = im.image.shape[0]
    width = im.image.shape[1]
    x = np.asarray(im.image)
    X_48.append(x)
    y = np.asarray(im.pyramid[0].label)
    Y_48.append(y)
X_48 = np.asarray(X_48)
Y_48 = np.asarray(Y_48)
X_48 = X_48.reshape(X_48.shape[0],1,X_48.shape[1], X_48.shape[2])
Y_48 = Y_48.reshape(Y_48.shape[0],1)
print("X_48-shape: {0}".format(X_48.shape))
print("Y_48-shape: {0}".format(Y_48.shape))

train.train48Net(X_48,Y_48,W_48, windowSize,  batchSize, nbEpoch)
