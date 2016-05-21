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
T12 = 0.3 #threshold for it is passed to 48net
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

print("=========================================")
#================================================
# 12 net
#================================================

print("\n\n============== 12Net ====================")
model12 = model_architecture.setUp12net(windowSize)
print("Loading model from: " + model12FileName)
model12.load_weights(model12FileName)

# Get best predictions
predictions_12 = model12.predict(X, batch_size=16, verbose=1)
print("12Net max prediction: {0}".format(np.max(predictions_12)))
print("12Net max label: {0}".format(np.max(Y)))
print("=========================================")

#================================================
# 48 net
#================================================

print("\n\n============== 48Net ====================")
prevWindowSize = windowSize
scaleFactor = 2
batchSize = 1
zoomFactor = 3

[X_48, Y_48, W_48, windowSize, imdb_48] = net.preProcess48Net(imdb, X, predictions_12,W, prevWindowSize, scaleFactor, zoomFactor, T12)

# Check if there are any face candidates at all from 12 net
if (X_48.shape[0]!= 0):
    model48 = model_architecture.setUp48net(windowSize)
    print("Loading model from: " + model48FileName)
    model48.load_weights(model48FileName)

    predictions_48 = model48.predict(X_48, batchSize, verbose=1)

    #================================================
    # Evaluation
    #================================================

    p_idx = np.reshape(0,1,np.asarray(predictions_48 > T48))
    t_idx = np.reshape(0,1,np.asarray(Y_48 > T48))
    print("\n========== Top predictions vs true ===========")
    print("Predicted max labels:\n{0}\nTrue labels:\n {1}".format(predictions_48[p_idx], Y_48[p_idx]))
    print("========== Top true vs predictions ===========")
    print("Max true labels:\n{1}\n predicted labels:\n {0}".format(predictions_48[p_idx], Y_48[p_idx]))
    print("==============================================")
    predicted_windows = W_48[p_idx,:,:]
    predicted_windows = predicted_windows.reshape(predicted_windows.shape[1],predicted_windows.shape[2])
    true_windows = W_48[t_idx,:,:]
    true_windows = true_windows.reshape(true_windows.shape[1],true_windows.shape[2])
    predicted_images = []
    true_images = []
    for i in range(0,len(predicted_windows)):
        predicted_images.append(imdb[predicted_windows[i][3]].image)
        true_images.append(np.copy(predicted_images[i]))
    predicted_images = np.asarray(predicted_images)
    true_images = np.asarray(true_images)

    predicted_subimages = X_48[p_idx,:,:]
    predicted_subimages = predicted_subimages.reshape(predicted_subimages.shape[1], predicted_subimages.shape[2], predicted_subimages.shape[3])

    true_subimages = X_48[t_idx,:,:]
    true_subimages = true_subimages.reshape(true_subimages.shape[1], true_subimages.shape[2], true_subimages.shape[3])

    vr.visualizeResult(predicted_images, predicted_subimages, predicted_windows, true_images, true_subimages, true_windows)
else:
    print("NO FACE!!!")
print("=========================================")
