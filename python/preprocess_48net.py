import numpy as np
from imagepyramid import ImagePyramid
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
            label = imdb[window[3]].pyramid[0].label

            #Generate new window, replace the old one, ## old => and calculate corresponding label for that window
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
                
                ### OLD ###
                ##Compare to window and calculate new label
                #labelwidth = label[2]
                #xlabel_left = label[0]-int(labelwidth/2)
                #xlabel_right = label[0]+int(labelwidth/2)
                #ylabel_upper = label[1]-int(labelwidth/2)
                #ylabel_lower = label[1]+int(labelwidth/2)
                #margin = 1.5/math.pow(labelwidth,2)
                #sublabelx = 1- margin*(math.pow(x1-xlabel_left,2)+ math.pow(x2-xlabel_right,2))
                #sublabelx = max(sublabelx, 0)
                #sublabely = 1- margin*(math.pow(y1-ylabel_upper,2)+ math.pow(y2-ylabel_lower,2))
                #sublabely = max(sublabely, 0)
                #sublabel = min(sublabelx, sublabely)

                #Shift original label to same coordinate system as subimage
                scaling = (image.shape[1]/(y2-y1)*1.0)
                newLabel = np.array(label)
                newLabel[0] = (label[0] - y1)/scaling
                newLabel[1] = (label[1] - x1)/scaling
                pyr = ImagePyramid(subImage, newLabel, scaleFactor, (newWindowSize,newWindowSize))
                imdb_48.append(pyr)
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
