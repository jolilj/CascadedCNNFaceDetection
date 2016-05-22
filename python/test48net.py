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

model12FileName = str(sys.argv[1])
model48FileName = str(sys.argv[2])
windowSize = 24
scaleFactor = 2
stepSize = 24
T48 = 0.3 #threshold for if it is a high label for the final evaluation

#================================================
# Preprocessing
#================================================

print("======Loading and Preprocessing...======")
start_time = time.time()

#If path to image passed as argument load and process that image, otherwise load from annotations(evaluation images)
if (len(sys.argv) > 3):
        imdb =[il.loadAndPreProcessSingle(str(sys.argv[3]), scaleFactor, (windowSize, windowSize))]
else:
    imdb = il.loadAndPreProcessIms('annotations_short.txt', scaleFactor, (windowSize,windowSize))

print("Image database: {0} images".format(len(imdb)))
[X, Y, W] = il.getCNNFormat(imdb, stepSize, windowSize)
print("finished preprocessing in {0}".format(time.time()-start_time))
print('X-shape')
print(X.shape)
print('Y-shape: ')
print(Y.shape)
print('W-shape')
print(W.shape)


# Get top 10%
targets  = np.squeeze(Y)
nb_top_targets = int(math.ceil(targets.shape[0]*0.1))
p_idx = np.argsort(targets)[-nb_top_targets:]
Y = Y[p_idx,:]
X = X[p_idx,:, :, :]
W = W[p_idx, :, :]
#print("=========================================")
##================================================
## 12 net
##================================================
#
#print("\n\n============== 12Net ====================")
#model12 = model_architecture.setUp12net(windowSize)
#print("Loading model from: " + model12FileName)
#model12.load_weights(model12FileName)
#
## Get best predictions
#predictions_12 = model12.predict(X, batch_size=16, verbose=1)
#print("12Net max prediction: {0}".format(np.max(predictions_12)))
#print("12Net max label: {0}".format(np.max(Y)))
#print("=========================================")

#================================================
# 48 net
#================================================

print("\n\n============== 48Net ====================")
prevWindowSize = windowSize
scaleFactor = 2
batchSize = 1
zoomFactor = 3

[X_48, Y_48, W_48, windowSize, imdb_48] = net.preProcess48Net(imdb, X, Y ,W, prevWindowSize)

### To map input to 48 with original image
#i = np.argmax(Y_48)
#y = Y_48[i,:]
#w = W_48[i,0,:]
#images = []
#for i in range(0,len(W_48)):
#    idx = W_48[i,0,3]
#    images.append(imdb[idx].image)

#images = np.squeeze(images)
#X_48 = np.squeeze(X_48)
#W_48 = np.squeeze(W_48)
#vr.visualizeResult(images, X_48, W_48)
model48 = model_architecture.setUp48net(windowSize)
print("Loading model from: " + model48FileName)
model48.load_weights(model48FileName)

predictions_48 = model48.predict(X_48, batchSize, verbose=1)

#================================================
# Evaluation
#================================================
p_idx = np.argmax(np.squeeze(predictions_48))
predicted_windows = W_48[p_idx,:,:]
predicted_images = []
for i in range(0,len(predicted_windows)):
    predicted_images.append(imdb[predicted_windows[i][3]].image)
predicted_images = np.asarray(predicted_images)

#predicted_subimages = X_48[p_idx,:,:]
#predicted_subimages = predicted_subimages.reshape(predicted_subimages.shape[1], predicted_subimages.shape[2], predicted_subimages.shape[3])

title = "Top predicted face image from 48Net"
vr.visualizeResultNoSubImage(title,predicted_images, predicted_windows)
print("=========================================")
