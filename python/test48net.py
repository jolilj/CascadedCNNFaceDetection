import math
from keras.models import Sequential
import model_architecture
import time
import imageloader as il
import sys
import imagepyramid
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import preprocess_48net as net
import visualize_results as vr

model48FileName = str(sys.argv[1])
windowSize48 = 48
windowSize12 = 24
stepSize = 24
scaleFactor = 2
batchSize = 1

#================================================
# Preprocessing
#================================================

print("======Loading and Preprocessing...======")
start_time = time.time()

#If path to image passed as argument load and process that image, otherwise load from annotations(evaluation images)
if (len(sys.argv) > 2):
        imdb =[il.loadAndPreProcessSingle(str(sys.argv[2]), scaleFactor, (windowSize12*4, windowSize12*4))]
else:
    imdb = il.loadAndPreProcessIms('annotations_short.txt', scaleFactor, (windowSize12*4,windowSize12*4))

print("Image database: {0} images".format(len(imdb)))
[X, Y, W] = il.getCNNFormat(imdb, stepSize, windowSize12)
print("finished preprocessing in {0}".format(time.time()-start_time))
print('X-shape')
print(X.shape)
print('Y-shape: ')
print(Y.shape)
print('W-shape')
print(W.shape)

#Load model
model48 = model_architecture.setUp48net(windowSize48)
print("Loading model from: " + model48FileName)
model48.load_weights(model48FileName)

# Get top 10%
targets  = np.squeeze(Y)
nb_top_targets = int(math.ceil(targets.shape[0]*0.1))
p_idx = np.argsort(targets)[-nb_top_targets:]
Y = Y[p_idx,:]
X = X[p_idx,:, :, :]
W = W[p_idx, :, :]

#================================================
# 48 net
#================================================

print("\n\n============== 48Net ====================")
start_time = time.time()
[X_48, Y_48, W_48, windowSize, imdb_48] = net.preProcess48Net(imdb, X, Y, W, windowSize12,0.5)
print("preprocessing in {0}".format(time.time()-start_time))
predictions_48 = model48.predict_on_batch(X_48)
print("prediction in {0} s".format(time.time()-start_time))
#================================================
# Evaluation
#================================================
## To map input to 48 with original image
i = np.argmax(predictions_48)
y = predictions_48[i,0]
w = W_48[i,0,:]
images = []
images.append(imdb[w[3]].image)
windows = []
windows.append(w)
title = "Top predicted face image from 48Net: {0:.3f}".format(y.item())
vr.visualizeResultNoSubImage(title,images, windows)
print("=========================================")
