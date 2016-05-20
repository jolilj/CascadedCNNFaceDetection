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
import visualizeResult as vr

model12FileName = str(sys.argv[1])
model48FileName = str(sys.argv[2])
windowSize = 24
scaleFactor = 2
stepSize = 24
T12 = 0.4 #threshold for it is passed to 48net
T48 = 0.6 #threshold for if it is a high label for the final evaluation

#================================================
# Preprocessing
#================================================

print("======Loading and Preprocessing...======")
start_time = time.time()

#If path to image passed as argument load and process that image, otherwise load from annotations(evaluation images)
if (len(sys.argv) > 3):
        imdb =[il.loadAndPreProcessSingle(str(sys.argv[3]), scaleFactor, (windowSize, windowSize))]
else:
    imdb = il.loadAndPreProcessIms('annotations_train_short.txt', scaleFactor, (windowSize,windowSize))

print("Image database: {0} images".format(len(imdb)))
[X, Y, W] = il.getCNNFormat(imdb, stepSize, windowSize)
print("finished preprocessing in {0}".format(time.time()-start_time))
print('X-shape')
print(X.shape)
print('Y-shape: ')
print(Y.shape)
print('W-shape')
print(W.shape)

#================================================
# 12 net
#================================================

model12 = model_architecture.setUp12net(windowSize)
print("Loading model from: " + model12FileName)
model12.load_weights(model12FileName)

# Get best predictions
predictions_12 = model12.predict(X, batch_size=16, verbose=1)


#================================================
# 48 net
#================================================

prevWindowSize = windowSize
scaleFactor = 2
batchSize = 1
zoomFactor = 3

[X_48, Y_48, W_48, windowSize, imdb_48] = net.preProcess48Net(imdb, X, predictions_12,W, prevWindowSize, scaleFactor, zoomFactor, T12)

model48 = model_architecture.setUp48net(windowSize)
print("Loading model from: " + model48FileName)
model48.load_weights(model48FileName)

predictions_48 = model48.predict(X_48, batchSize, verbose=1)

#================================================
# Evaluation
#================================================


p_idx = np.squeeze(predictions_48 > T48)
t_idx = np.squeeze(Y_48 > T48)

print("Predicted max labels:\n{0}\nlabel: {1}".format(predictions_48[p_idx], Y_48[p_idx]))
print("Predicted label:\n{0}\nmax label: {1}".format(predictions_48[t_idx], Y_48[t_idx]))
predicted_windows = np.squeeze(W_48[p_idx,:,:])
true_windows = np.squeeze(W_48[t_idx,:,:])

predicted_images = []
true_images = []
for i in range(0,len(predicted_windows)):
    predicted_images.append(imdb[predicted_windows[i][3]].image)
    true_images.append(np.copy(predicted_images[i]))

predicted_subimages = X_48[p_idx,:,:]
predicted_subimages = np.squeeze(predicted_subimages)

true_subimages = X_48[t_idx,:,:]
true_subimages = np.squeeze(true_subimages)


vr.visualizeResult(predicted_images, predicted_subimages, predicted_windows, true_images, true_subimages, true_windows)







