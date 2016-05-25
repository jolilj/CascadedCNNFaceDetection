##############################
# Entry script for testing
# Face Detection using 
# Cascaded Convolutional Networks
#############################
# This script loads a set of images 
# as annotated in txt file annotated_test,
# randomly picks 5 of them
# and plots each image with its predicted
# face bounding box, one at the time
#############################

from keras.models import Sequential
import model_architecture
import time
import imageloader as il
import math
import sys
import imagepyramid
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import preprocess_edge_48net as net
import visualize_results as vr
import compute_accuracy as acc
import random

model24FileName = '24_trained_model_final.hp5'
model48FileName = '48_trained_model_final.hp5'
windowSize = 24
windowSize48 = 48
scaleFactor = 1.5# This parameter can be tweaked to yield different results
stepSize = 12 # This parameter can be tweaked to yield different results
imNum = 5
#================================================
# Loading and  Preprocessing
#================================================

print("======Loading and Preprocessing...======")
start_time = time.time()
#Load images from databas
imdb = il.loadAndPreProcessIms('annotations_test_short.txt', scaleFactor, (windowSize,windowSize))
rand_idx = random.sample(range(0, len(imdb)), imNum)
imdb_rand=[]
print(rand_idx)
for i in rand_idx:
    imdb_rand.append(imdb[i])
imdb = imdb_rand
print("Image database: {0} images".format(len(imdb)))
[X, Y, W] = il.getCNNFormat(imdb, stepSize, windowSize)
print("finished preprocessing in {0}".format(time.time()-start_time))
print('X-shape')
print(X.shape)
print('Y-shape: ')
print(Y.shape)
print('W-shape')
print(W.shape)
print("=========================================")


#================================================
# 24 net
#================================================

print("\n\n============== 24Net ====================")
model24 = model_architecture.setUp12net(windowSize)
print("Loading model from: " + model24FileName)
model24.load_weights(model24FileName)

model48 = model_architecture.setUp48net(windowSize48)
print("Loading model from: " + model48FileName)
model48.load_weights(model48FileName)

#==================Run through 24Net==================
#=====================================================
start_time = time.time()
#=== Predict
predictions_24 = model24.predict(X, batch_size=16, verbose=1)
print("24Net detection in {0} s".format(time.time() - start_time))
#====
predictions_high = []
previmageidx = W[0][0][3]
X_per_image = []
W_per_image = []
pred_per_image = []
X_high = []
W_high = []

for i in range(0, predictions_24.shape[0]):
    imageidx = W[i][0][3]
    if(imageidx == previmageidx):
        pred_per_image.append(predictions_24[i, :])
        X_per_image.append(X[i, :, :, :])
        W_per_image.append(W[i, :, :])
    else:
        nb_top_targets = int(math.ceil(len(pred_per_image)*0.1))
        high_idx = np.argsort(np.squeeze(pred_per_image))[-nb_top_targets:]
        for index in high_idx:
                X_high.append(X_per_image[index][:][:][:])
                W_high.append(W_per_image[index][:][:])
                predictions_high.append(pred_per_image[index][:])
        X_per_image = []
        W_per_image = []
        pred_per_image = []
    previmageidx = imageidx

X_high = np.asarray(X_high)
W_high = np.asarray(W_high)
predictions_high = np.asarray(predictions_high)

#=====================================================
#=====================================================


##================================================
# 48 net
#================================================

print("\n\n============== 48Net ====================")
start_time = time.time()
[X_48, Y_48, W_48, windowSize, imdb_48] = net.preProcess48Net(imdb, X_high, predictions_high, W_high, windowSize)
print("preprocessing in {0}".format(time.time()-start_time))
predictions_48 = model48.predict_on_batch(X_48)
print("48 detecion in {0} s".format(time.time()-start_time))
#================================================
# Evaluation
#================================================
## To map input to 48 with original image
maxlabels = []
previmageidx = W_48[0][0][3]
maxlabel = -1
label = -1
maxindex = -1

for i in range(0, W_48.shape[0]):
    imageidx = W_48[i][0][3]
    if(imageidx == previmageidx):
        label = predictions_48[i][:]
        if (maxlabel < label):
            maxlabel = label
            maxindex = i
    else:
        maxlabels.append([maxlabel,maxindex])
        maxlabel = 0
        previmageidx = imageidx
maxlabels.append([maxlabel,maxindex])

maxlabels = np.asarray(maxlabels)
#===========================
# Evaluation for a data set
#===========================

W_48 = np.squeeze(W_48)
windows = []
images = []

for i in range(maxlabels.shape[0]):
    maxindex = int(maxlabels[i][1])
    imageidx = W_48[maxindex,3]
    windows.append(W_48[maxindex,:])
    images.append(imdb[imageidx].image)

windows = np.asarray(windows)

similarity = acc.compute_accuracy_dataset(maxlabels, imdb, W_48)
thres_acc = 0
tracked = 0

for a in similarity:
    if a>thres_acc:
        tracked = tracked+1

prec = float(tracked)/float(len(similarity))

print("Accuracy: {0}".format(prec))

title = "Top predicted face images from 48Net"
vr.visualizeResultNoSubImage(title,images, windows)
