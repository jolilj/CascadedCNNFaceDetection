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


##Visualize the result in a figure
#   *pImages - the predicted images
#   *pSubImages - the predicted subimages
#   *pWindows - the corresponding window of the subimages
#   *tImages - true images
#   *tSubImages - true subimages
#   *tWindows - true windows

def visualizeResult(pImages, pSubImages, pWindows, tImages, tSubImages, tWindows):
    for i in range(0,len(pImages)):
        pImage = pImages[i]
        pSubImage = pSubImages[i]
        pWindow = pWindows[i]

        predicted_rect = imagepyramid.labelToRect(pWindow)
        cv2.rectangle(pImage, predicted_rect[1], predicted_rect[0], [0, 255, 0],1 )

        fig = plt.figure()
        fig.add_subplot(2,2,1)
        plt.imshow(pImage,cmap=plt.cm.gray)
        fig.add_subplot(2,2,2)
        plt.imshow(pSubImage, cmap=plt.cm.gray)

        #if targets are available
        if (i < len(tImages)):
            tImage = tImages[i]
            tSubImage = tSubImages[i]
            tWindow = tWindows[i]

            true_rect =  imagepyramid.labelToRect(tWindow)
            cv2.rectangle(tImage, true_rect[1], true_rect[0], [0, 255, 0],1 )

            fig.add_subplot(2,2,3)
            plt.imshow(tImage,cmap=plt.cm.gray)
            fig.add_subplot(2,2,4)
            plt.imshow(tSubImage, cmap=plt.cm.gray)
        plt.show()

#################################################
#################################################
############## SCRIPT STARTS ####################
#################################################
#################################################
modelFileName = str(sys.argv[1])
windowSize = 128
scaleFactor = 1.5
stepSize = 16
T = 0.85

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

# Get best predictions
predictions = model.predict(X, batch_size=16, verbose=1)
p_idx = np.squeeze(predictions > T)
t_idx = np.squeeze(Y > T)

print("Predicted max labels:\n{0}\nlabel: {1}".format(predictions[p_idx], Y[p_idx]))
print("Predicted label:\n{0}\nmax label: {1}".format(predictions[t_idx], Y[t_idx]))
print(p_idx.shape)
predicted_windows = np.squeeze(W[p_idx,:,:])
true_windows = np.squeeze(W[t_idx,:,:])

predicted_images = []
true_images = []
for i in range(0,len(predicted_windows)):
    predicted_images.append(imdb[predicted_windows[i][3]].image)
    true_images.append(np.copy(predicted_images[i]))

predicted_subimages = X[p_idx,:,:]
predicted_subimages = np.squeeze(predicted_subimages)

true_subimages = X[t_idx,:,:]
true_subimages = np.squeeze(true_subimages)

visualizeResult(predicted_images, predicted_subimages, predicted_windows, true_images, true_subimages, true_windows)

