import numpy as np
from imagepyramid import ImagePyramid
import math

def preProcess48Net(imdb, X,Y,W, prevWindowSize, scaleFactor, zoomFactor, T=0.5):
    idx = np.squeeze(Y>T)
    valid_windows = W[idx,:,:]
    imdb_48 = []
    for i in range(0,valid_windows.shape[0]):
        window = np.squeeze(valid_windows[i,:,:])
        pyr = imdb[window[3]]
        image = pyr.image
        label = pyr.pyramid[0].label
        valid_windows[i,:,2] = window[2]*zoomFactor
        y1 = int(window[1]-prevWindowSize*zoomFactor/2)
        y2 = int(window[1]+prevWindowSize*zoomFactor/2)
        x1 = int(window[0]-prevWindowSize*zoomFactor/2)
        x2 = int(window[0]+prevWindowSize*zoomFactor/2)
        if (not (y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1])):  
            subImage = image[y1:y2,x1:x2]
            print(subImage.shape)
            windowSize = prevWindowSize*zoomFactor
            labelwidth = label[2]
            xlabel_left = label[0]-int(labelwidth/2)
            xlabel_right = label[0]+int(labelwidth/2)
            ylabel_upper = label[1]-int(labelwidth/2)
            ylabel_lower = label[1]+int(labelwidth/2)

            #Compare to window and calculate new label
            margin = 1.5/math.pow(labelwidth,2)
            sublabelx = 1- margin*(math.pow(x1-xlabel_left,2)+ math.pow(x2-xlabel_right,2))
            sublabelx = max(sublabelx, 0)
            sublabely = 1- margin*(math.pow(y1-ylabel_upper,2)+ math.pow(y2-ylabel_lower,2))
            sublabely = max(sublabely, 0)
            sublabel = min(sublabelx, sublabely)

            imdb_48.append(ImagePyramid(subImage, sublabel, scaleFactor, (windowSize/4,windowSize/4)))

    print("===VALID IMS====")
    X_48 = []
    Y_48 = []
    W_48 = valid_windows
    for im in imdb_48:
        height = im.image.shape[0]
        width = im.image.shape[1]
        x = np.asarray(im.image)
        X_48.append(x)
        y = np.asarray(im.pyramid[0].label)
        Y_48.append(y)
    X_48 = np.asarray(X_48)
    Y_48 = np.asarray(Y_48)
    X_48 = X_48.reshape(X_48.shape[0],1,X_48.shape[1], X_48.shape[2])
    Y_48 = Y_48.reshape(Y_48.shape[0],1)
    print("X_48-shape: {0}".format(X_48.shape))
    print("Y_48-shape: {0}".format(Y_48.shape))

    return [X_48, Y_48, W_48, windowSize]