import numpy as np
from imagepyramid import ImagePyramid
import matplotlib.pyplot as plt
import math
import cv2
import imageloader as il
def preProcess48Net(imdb, X,Y,W, prevWindowSize, newWindowSize, scaleFactor, zoomFactor, T=0.5):
    #Get all indices where the output is higher than the threshold
    idx = np.squeeze(Y>T)
    W_48 = W[idx,:,:]
    print("Number of subimages above treshold: {0}".format(len(W_48)))
    print("Valid Windows: {0}".format(W_48.shape))
    imdb_48 = []
    wIdx = []
    if (W_48.shape[0] != 0):
        for i in range(0,W_48.shape[0]):
            window = np.squeeze(W_48[i,:,:])
            # Get corresponding image to crop new subimages
            image = imdb[window[3]].image
            # Get the original label
            label = imdb[window[3]].pyramid[0].label

            #Generate new bigger window, replace the old one
            W_48[i,:,2] = window[2]*zoomFactor
            y1 = int(window[1]-prevWindowSize*zoomFactor/2)
            y2 = int(window[1]+prevWindowSize*zoomFactor/2)
            x1 = int(window[0]-prevWindowSize*zoomFactor/2)
            x2 = int(window[0]+prevWindowSize*zoomFactor/2)

            # Discard edge windows (wrong size...lets hope the face isnt at the edge xD)
            if (not (y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1])):
                wIdx.append(True)
                # Crop the image
                subImage = image[y1:y2,x1:x2]
                subImageSize = prevWindowSize*zoomFactor

                #Shift original label to same coordinate system as subimage
                #scaling = (image.shape[1]/(y2-y1)*1.0)
                print("===Coordinates for window===")
                print("x1: {0}, y1: {1}, w: {2}".format(x1,y1, subImageSize))
                x = int(label[0] - label[2]/2)
                y = int(label[1] - label[2]/2)
                print("===Coordinates for label===")
                print("x: {0}, y: {1}, w: {2}".format(x,y, label[2]))
                newLabel = []
                newLabel.append(int(x - x1 + label[2]/2))
                newLabel.append(int(y-y1 + label[2]/2))
                newLabel.append(label[2])
                newLabel = np.asarray(newLabel)
                print("===New label ===")
                print("x: {0}, y: {1}".format(newLabel[0], newLabel[1]))
                pyr = ImagePyramid(subImage, newLabel, scaleFactor, (newWindowSize,newWindowSize))
                imdb_48.append(pyr)
                #title = ("label:  {0:.2f}").format(sublabel)
                title = "hm"
                fig = plt.figure(title)
                fig.add_subplot(1,2,1)
                copy = image.copy()
                cv2.rectangle(copy, (int(x),int(y)), (int(x+label[2]), int(y+label[2])), [0, 255, 0],1 )
                plt.imshow(copy,cmap=plt.cm.gray)
                fig.add_subplot(1,2,2)
                plt.imshow(subImage, cmap=plt.cm.gray)
                plt.show()

            else:
                wIdx.append(i)
        wIdx = np.asarray(wIdx)
        # Delete windows for edge images from W_48
        W_48 = W_48[wIdx,:,:]

    print("imdb length: {0}".format(len(imdb_48))) 
    print("Length of pyramid: {0}".format(len(imdb_48[0].pyramid)))
    [X_48, Y_48, W_48] = il.getCNNFormatSingle(imdb_48, newWindowSize)

    X_48 = np.asarray(X_48)
    Y_48 = np.asarray(Y_48)
    print("X_48-shape: {0}".format(X_48.shape))
    print("Y_48-shape: {0}".format(Y_48.shape))
    print("W_48-shape: {0}".format(W_48.shape))

    return [X_48, Y_48, W_48, newWindowSize, imdb_48]
