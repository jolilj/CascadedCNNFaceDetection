from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten
from keras.optimizers import SGD
import imageloader as il
import slidingwindow as sw
import numpy as np
import cv2
import time
import model_architecture
import os

windowSize = 128
scaleFactor = 1.5
stepSize = 16
modelFileName = 'trained_model_w' + str(windowSize) + '_scale' + str(scaleFactor) + '_step' + str(stepSize) + '.h5'

print("======Loading and Preprocessing...======")
start_time = time.time()
imdb = il.loadAndPreProcessIms('annotations_short.txt', scaleFactor, (windowSize,windowSize))

[X, Y, W] = il.getCNNFormat(imdb, stepSize, windowSize)
print("finished preprocessing in {0}".format(time.time()-start_time))
print('X-shape')
print(X.shape)
print('Y-shape: ')
print(Y.shape)
model = model_architecture.setUpModel(windowSize)
if (os.path.exists(os.getcwd()+'/' + modelFileName)):
    model.load_weights(modelFileName)
    print("Loaded model: " + modelFileName)

else:
    print("No model stored, create new")

print("======Training....======")
model.fit(X, Y, batch_size=16, nb_epoch=4, verbose=1)
print("Finished training!")
model.save_weights(modelFileName, overwrite=True)
print("saved model to: " + modelFileName) 


#copy = X[maxIdx[0]][maxIdx[1]][maxIdx[2]].copy()
#cv2.imshow('sub', copy)
#cv2.waitKey(0)


