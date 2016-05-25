import numpy as np
import cv2
import time
import math
from imagepyramid import ImagePyramid
from PIL import Image
#import matplotlib.pyplot as plt

## Defines a sliding window that is sliding over the image cascade
# Returns a list [windows, labels, croppedImages] where
#   * windows[i] are the positions of all slided windows of imagePyramid.pyramid[i]
#   * labels[i] are all calculated labels -------------------||-------------------
#   * croppedImages[i] are the corresponding cropped images

def slideWindow(imagePyramid, imageIdx, stepSize, windowSize):
	# labels are stored as [[cx,cy,w], [cx,cy,w]....]
	croppedImages = []
	windows = []
	labels = []

        #loop through all images in pyramid and do a sliding window for each of them
	for i in range(0, len(imagePyramid.pyramid)):
                scaleFactor = math.pow(imagePyramid.scale,i)
		image = imagePyramid.pyramid[i].image
		label = imagePyramid.pyramid[i].label
		image_width = image.shape[1]
		image_height = image.shape[0]
                if (windowSize <= min(image_width, image_height)):
                    sublabels = []
                    subImages = []
                    windowPositions = []
                    y_range = image_height-windowSize
                    if (y_range == 0):
                        y_range = 1
                    x_range = image_width-windowSize
                    if (x_range == 0):
                        x_range = 1
                    for y in range(0,y_range,stepSize):
                            for x in range(0,x_range,stepSize):
                                    subImage = image[y:y+windowSize,x:x+windowSize]

                                    #Store window position along with it's corresponding image index (for mapping window to image)
                                    windowInfo = [int((x+windowSize/2)*scaleFactor),int((y+windowSize/2)*scaleFactor), int(windowSize*scaleFactor), imageIdx]
                                    windowPositions.append(windowInfo)
                                    subImages.append(subImage)
                                    #Get image label if available
                                    if (type(label) != int):
                                        labelwidth = label[2]
                                        xlabel_left = int(label[0]-labelwidth/2)
                                        xlabel_right = int(label[0]+labelwidth/2)
                                        ylabel_upper = int(label[1] - labelwidth/2)
                                        ylabel_lower = int(label[1]+labelwidth/2)

                                        #Compare to window and calculate new label
                                        margin = 2/math.pow(labelwidth,2)
                                        sublabelx = 1- margin*(math.pow(x-xlabel_left,2)+ math.pow(x+windowSize-xlabel_right,2))
                                        sublabelx = max(sublabelx, 0.0)
                                        sublabely = 1- margin*(math.pow(y-ylabel_upper,2)+ math.pow(y+windowSize-ylabel_lower,2))
                                        sublabely = max(sublabely, 0.0)
                                        sublabel = min(sublabelx, sublabely)
                                    else:
                                        sublabel = label
                                    
                                    
                                    #Append to image's sublabels
                                    sublabels.append(sublabel)
				    
                                    

                    #Append image's sublabels to the new labels
                    labels.append(sublabels)
                    croppedImages.append(subImages)
                    windows.append(windowPositions)
        return [windows, labels, croppedImages]


