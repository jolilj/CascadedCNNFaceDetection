from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten
from keras.optimizers import SGD
from tempfile import TemporaryFile
import imageloader as il
import slidingwindow as sw
import numpy as np
import cv2
import time
import model_architecture
import os
import sys
import train12Net as train

windowSize = 24
scaleFactor = 2
stepSize = 24
batchSize = 16
nbEpoch = 10
modelFileName = '12_trained_model_w' + str(windowSize) + '_scale' + str(scaleFactor) + '_step' + str(stepSize) + '.h5'

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
    imdb = il.loadAndPreProcessIms('annotations_train_short.txt', scaleFactor, (windowSize,windowSize))

    [X, Y, W] = il.getCNNFormat(imdb, stepSize, windowSize)
    np.save('data/data_X',X)
    np.save('data/data_Y',Y)
    np.save('data/data_W',W)
    print("finished preprocessing in {0}".format(time.time()-start_time))


print("X-shape: {0}".format(X.shape))
print("Y-shape: {0}".format(Y.shape))

train.train12Net(X,Y, windowSize, scaleFactor, stepSize, batchSize, nbEpoch)

