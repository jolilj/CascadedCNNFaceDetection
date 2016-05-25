import numpy as np
from imagepyramid import ImagePyramid
import matplotlib.pyplot as plt
import math
import cv2
import imageloader as il

newWindowSize = 48
stepSize = 24
batchSize = 16
nbEpoch = 100
zoomFactor = 5
pyramidLevels = 3

# Preprocessing the high predictions from the 24-net for the 48-net



def preProcess48Net(imdb, X,Y,W, prevWindowSize, T=0):
    scaleFactor = math.pow(prevWindowSize*zoomFactor*1.0/newWindowSize,1.0/(pyramidLevels-1))
    #Get all indices where the output is higher than the threshold
    idx = np.squeeze(Y>T)
    valid_windows = W[idx,:,:]
    #print("Number of subimages above threshold: {0}".format(len(valid_windows)))
    #print("Valid Windows: {0}".format(valid_windows.shape))
    imdb_48 = []
    windowPos = []
    if (valid_windows.shape[0] != 0):
        for i in range(0,valid_windows.shape[0]):
            window = np.squeeze(valid_windows[i,:,:])
            # Get corresponding image to crop new subimages
            image = imdb[window[3]].image
            # Get the original label
            label = imdb[window[3]].pyramid[0].label
            #Generate new bigger window
            y1 = int(window[1]-prevWindowSize*zoomFactor/2)
            y2 = int(window[1]+prevWindowSize*zoomFactor/2)
            x1 = int(window[0]-prevWindowSize*zoomFactor/2)
            x2 = int(window[0]+prevWindowSize*zoomFactor/2)

            # Discard edge windows (wrong size...lets hope the face isnt at the edge xD)
            if (not (y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1])):
                windowPos.append([x1,y1, window[3]])
                # Crop the image
                subImage = image[y1:y2,x1:x2]
                subImageSize = prevWindowSize*zoomFactor
                newLabel = label
                if (type(label) != int):
                    #Shift original label to same coordinate system as subimage
                    x = int(label[0] - label[2]/2)
                    y = int(label[1] - label[2]/2)
                    newLabel = []
                    newLabel.append(int(x - x1 + label[2]/2))
                    newLabel.append(int(y-y1 + label[2]/2))
                    newLabel.append(label[2])
                    newLabel = np.asarray(newLabel)
                    
                pyr = ImagePyramid(subImage, newLabel, scaleFactor, (newWindowSize,newWindowSize))
                imdb_48.append(pyr)
                
            else:
                print("discarded image")
    [X_48, Y_48, W_48] = il.getCNNFormatSingle(imdb_48, newWindowSize, windowPos)

    X_48 = np.asarray(X_48)
    Y_48 = np.asarray(Y_48)
    
    return [X_48, Y_48, W_48, newWindowSize, imdb_48]
