from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten
from keras.optimizers import SGD
import imageloader as il
import slidingwindow as sw
import numpy as np
import cv2
import time


windowSize = 128
scaleFactor = 1.5
stepSize = 4

imdb = il.loadAndPreProcessIms('annotations_short.txt', scaleFactor, (windowSize,windowSize))

X_organized = []
Y_organized = []
X = []
Y = []
for image in imdb:
	[windows, labels, croppedImages] = sw.slideWindow(image, stepSize, windowSize)

	X_organized.append(croppedImages)
	Y_organized.append(labels)
	for i in range(0,len(croppedImages)):
		for j in range(0,len(croppedImages[i])):
			X.append(croppedImages[i][j])
			Y.append(labels[i][j])


X = np.asarray(X)
X = X.reshape(X.shape[0],1,X.shape[1], X.shape[2])
print("X - shape:")
print(X.shape)
Y = np.asarray(Y)
Y=Y.reshape(Y.shape[0],1)
print("Y - shape:")
print(Y.shape)
model = Sequential()
model.add(Convolution2D(16, 3, 3, border_mode='valid', input_shape=(1, windowSize, windowSize)))
#model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
#model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1))


#maxLabel = 0
#maxIdx = -1
#for i in range(0,len(Y)):
#	for j in range(0,len(Y[i])):
		#for k in range(0,len(Y[i][j])):
#		subLabels = Y[i][j]
#		m = max(subLabels)			
#		index = np.argmax(subLabels)
#		if (m > maxLabel):
#			maxLabel = m
#			maxIdx = [i,j,index]
#print(maxLabel)	

#copy = X[maxIdx[0]][maxIdx[1]][maxIdx[2]].copy()
#cv2.imshow('sub', copy)
#cv2.waitKey(0)

sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=100)
print("======Compiling....======")
start_time = time.time()
model.compile(loss='mean_squared_error', optimizer=sgd)
print("finished compiling in {0}".format(time.time()-start_time))
print("======Training....======")
model.fit(X, Y, batch_size=16, nb_epoch=10, verbose=1)
print("Finished training!")
