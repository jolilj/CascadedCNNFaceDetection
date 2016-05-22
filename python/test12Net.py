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
import visualize_results as vr

#################################################
#################################################
############## SCRIPT STARTS ####################
#################################################
#################################################
modelFileName = str(sys.argv[1])
windowSize = 24
scaleFactor = 1.5
stepSize = 24
T = 0.5

print("======Loading and Preprocessing...======")
start_time = time.time()

#If path to image passed as argument load and process that image, otherwise load from annotations(evaluation images)
if (len(sys.argv) > 2):
        imdb =[il.loadAndPreProcessSingle(str(sys.argv[2]), scaleFactor, (windowSize, windowSize))]
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

model = model_architecture.setUp12net(windowSize)
print("Loading model from: " + modelFileName)
model.load_weights(modelFileName)

# Get top 10% predictions
predictions = model.predict(X, batch_size=16, verbose=1)
predictions = np.squeeze(predictions)
nb_top_predictions = int(math.ceil(predictions.shape[0]*0.1))
p_idx = np.argsort(predictions)[-nb_top_predictions:]
predictions = predictions[p_idx]

predicted_windows = np.squeeze(W[p_idx,:,:])
predicted_images = []
for i in range(0,len(predicted_windows)):
    predicted_images.append(imdb[predicted_windows[i][3]].image)

predicted_subimages = X[p_idx,:,:]
predicted_subimages = np.squeeze(predicted_subimages)

title = "Top 10%"
vr.visualizeResultNoSubImage(title,predicted_images, predicted_windows)

