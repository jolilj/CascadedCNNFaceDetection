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



def train48Net(X,Y,W, windowSize=48, batchSize=16, nbEpoch=10):
    #Load model architecture
    modelFileName = '48_trained_model_w' + str(windowSize) + '_batch' + str(batchSize) +'.h5'
    model = model_architecture.setUp48net(windowSize)
    if (os.path.exists(os.getcwd()+'/' + modelFileName)):
        model.load_weights(modelFileName)
        print("Loaded model: " + modelFileName)

    else:
        print("No model stored, creating new")

    print("======Training....======")
    history = model.fit(X, Y, batch_size=batchSize, nb_epoch=nbEpoch, verbose=1)
    print("Finished training!")
    model.save_weights(modelFileName, overwrite=True)
    print("saved model to: " + modelFileName) 
    return history
