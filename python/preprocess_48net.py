import numpy as np
from imagepyramid import ImagePyramid
import math
import cv2
import imageloader as il
def preProcess48Net(imdb, X,Y,W, prevWindowSize, scaleFactor, zoomFactor, T=0.5):
    #Get all indices where the output is higher than the threshold
    idx = np.squeeze(Y>T)
    W_48 = W[idx,:,:]
    print("Number of subimages above treshold: {0}".format(len(W_48)))
    print("Valid Windows: {0}".format(W_48.shape))
    imdb_48 = []
    wIdx = []
    windowSize = 0
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
                windowSize = prevWindowSize*zoomFactor
                
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
                print("oldlabel: {0}".format(label))
                scaling = (image.shape[1]/(y2-y1)*1.0)
                print("x1: {0}, y1: {1}".format(x1,y1))
                print(scaling)
                newLabel = np.array(label)
                newLabel[0] = (label[0] - y1)/scaling
                newLabel[1] = (label[1] - x1)/scaling
                print("newlabel: {0}".format(newLabel))
                print("scaleFactor {0}".format(scaleFactor))
                pyr = ImagePyramid(subImage, newLabel, scaleFactor, (windowSize/4,windowSize/4))
                print("Length of pyramid: {0}".format(len(pyr.pyramid)))
                imdb_48.append(pyr)
            else:
                wIdx.append(i)
        wIdx = np.asarray(wIdx)
        # Delete windows for edge images from W_48
        W_48 = W_48[wIdx,:,:]


    ### FOR DEBUGGIN, CHECK ALL PYRAMID IMAGES => YES THEY ARE CORRECT, PROBLEM IN SLIDING WINDOW!!! #####
    for im in imdb_48[0].pyramid:
        x = im.image
        title = ("subimage")
        copy = x.copy()
        #cv2.rectangle(copy, (x,y), (x+windowSize, y+windowSize), [255, 255, 255],1 )
        cv2.imshow(title, copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("windowsize: {0}".format(windowSize))
    print("image: {0}".format(imdb_48[0].image.shape))
    [X_48, Y_48, W_48] = il.getCNNFormatSingle(imdb_48, windowSize)
    for i in range(0,X_48.shape[0]):
        x = np.squeeze(X_48[i,:,:,:])
        y = np.squeeze(Y_48[i,:])
        title = ("subimage cnn format")
        copy = x.copy()
        #cv2.rectangle(copy, (x,y), (x+windowSize, y+windowSize), [255, 255, 255],1 )
        cv2.imshow(title, copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    X_48 = np.asarray(X_48)
    Y_48 = np.asarray(Y_48)
    if (X_48.shape[0] != 0):
        X_48 = X_48.reshape(X_48.shape[0],1,X_48.shape[1], X_48.shape[2])
        Y_48 = Y_48.reshape(Y_48.shape[0],1)
    print("X_48-shape: {0}".format(X_48.shape))
    print("Y_48-shape: {0}".format(Y_48.shape))
    print("W_48-shape: {0}".format(W_48.shape))

    return [X_48, Y_48, W_48, windowSize, imdb_48]
