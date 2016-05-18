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

modelFileName = str(sys.argv[1])
windowSize = 64
scaleFactor = 1.5
stepSize = 8

print("======Loading and Preprocessing...======")
start_time = time.time()
imdb = il.loadAndPreProcessIms('annotations_short.txt', scaleFactor, (windowSize,windowSize))

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

predictions = model.predict(X, batch_size=16, verbose=1)
i = np.argmax(predictions)
window = W[i,:,:]
window = np.append(window,[windowSize])
print("Max window: ")
print(window)
image = X[i,:,:]
image = np.squeeze(image)

print("Image: ")
print(image.shape)
rect = imagepyramid.labelToRect(window)
cv2.rectangle(image, rect[0], rect[1], [0, 255, 0],1 )

plt.imshow(image,cmap=plt.cm.gray)
plt.show()

#print("Predictions: | Truth: ")
#for i in range(0,len(predictions)):
#        print(str(predictions[i]) + "  |  " + str(Y[i]) )
