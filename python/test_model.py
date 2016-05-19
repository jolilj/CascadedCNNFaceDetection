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
#   *pImage - the predicted image
#   *pSubImage - the predicted subimage
#   *pWindow - the corresponding window of the subimage
#   *tImage - true image
#   *tSubImage - true subimage
#   *tWindow - true window

def visualizeResult(pImage, pSubImage, pWindow, tImage, tSubImage, tWindow):

    predicted_rect = imagepyramid.labelToRect(pWindow)
    true_rect =  imagepyramid.labelToRect(tWindow)

    cv2.rectangle(pImage, predicted_rect[1], predicted_rect[0], [0, 255, 0],1 )
    cv2.rectangle(tImage, true_rect[1], true_rect[0], [0, 255, 0],1 )

    fig = plt.figure()
    fig.add_subplot(2,2,1)
    plt.imshow(predicted_image,cmap=plt.cm.gray)
    fig.add_subplot(2,2,2)
    plt.imshow(predicted_subimage, cmap=plt.cm.gray)

    fig.add_subplot(2,2,3)
    plt.imshow(true_image,cmap=plt.cm.gray)
    fig.add_subplot(2,2,4)
    plt.imshow(true_subimage, cmap=plt.cm.gray)
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

model = model_architecture.setUpModel(windowSize)
print("Loading model from: " + modelFileName)
model.load_weights(modelFileName)

# Just to test, take the best prediction (faciest face....)
predictions = model.predict(X, batch_size=16, verbose=1)
i = np.argmax(predictions)
y = np.argmax(Y)

print("Predicted max label: {0}, label: {1}".format(predictions[i], Y[i]))
print("Predicted label: {0}, max label: {1}".format(predictions[y], Y[y]))
print("Correct label? {0}".format(i==y))

predicted_window = np.squeeze(W[i,:,:])
true_window = np.squeeze(W[y,:,:])

predicted_image = imdb[predicted_window[3]].image
true_image = np.copy(predicted_image) 

predicted_subimage = X[i,:,:]
predicted_subimage = np.squeeze(predicted_subimage)

true_subimage = X[y,:,:]
true_subimage = np.squeeze(true_subimage)

visualizeResult(predicted_image, predicted_subimage, predicted_window, true_image, true_subimage, true_window)

