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

def slideWindow(imagePyramid, stepSize, windowSize):
	# labels are stored as [[cx,cy,w], [cx,cy,w]....]
	croppedImages = []
	windows = []
	labels = []

        #loop through all images in pyramid and do a sliding window for each of them
	for i in range(0, len(imagePyramid.pyramid)):
		image = imagePyramid.pyramid[i].image
		label = imagePyramid.pyramid[i].label
		image_width = image.shape[1]
		image_height = image.shape[0]

                if (windowSize <= min(image_width, image_height)):
                    sublabels = []
                    subImages = []
                    windowPositions = []
                    for y in np.arange(0,image_height-windowSize,stepSize):
                            for x in np.arange(0,image_width-windowSize,stepSize):
                                    windowCenter = [x+int(windowSize/2),y+int(windowSize/2)]
                                    
                                    #Return 
                                    subImage = image[y:y+windowSize,x:x+windowSize]
                                    windowPositions.append(windowCenter)
                                    subImages.append(subImage)

                                    #Get image label
                                    labelwidth = label[2]
                                    xlabel_left = label[0]-int(labelwidth/2)
                                    xlabel_right = label[0]+int(labelwidth/2)
                                    ylabel_upper = label[1]-int(labelwidth/2)
                                    ylabel_lower = label[1]+int(labelwidth/2)

                                    #Compare to window and calculate new label
                                    margin = 1.5/math.pow(labelwidth,2)
                                    sublabelx = 1- margin*(math.pow(x-xlabel_left,2)+ math.pow(x+windowSize-xlabel_right,2))
                                    sublabelx = max(sublabelx, 0)
                                    sublabely = 1- margin*(math.pow(y-ylabel_upper,2)+ math.pow(y+windowSize-ylabel_lower,2))
                                    sublabely = max(sublabely, 0)
                                    sublabel = min(sublabelx, sublabely)
                                    
                                    #Append to image's sublabels
                                    sublabels.append(sublabel)
				    #title=str(subImage.shape)
                                    #title = ("label:  {0:.2f}").format(sublabel)
                                    #copy = image.copy()
                                    #cv2.rectangle(copy, (x,y), (x+windowSize, y+windowSize), [255, 255, 255],1 )
                                    #cv2.imshow(title, copy)
				    #cv2.imshow('sub',subImage)
                                    #cv2.waitKey(0)
                                    #cv2.destroyAllWindows()
                                    #time.sleep(0.03)

                    #Append image's sublabels to the new labels
                    labels.append(sublabels)
                    croppedImages.append(subImages)
                    windows.append(windowPositions)
        return [windows, labels, croppedImages]

## Test the implementation

#pil_img = Image.open('images/2002/07/19/big/img_581.jpg').convert('L')
#img = np.array(pil_img)


#imagePyramid = ImagePyramid(img, np.asarray([155.093404, 189.450662, 205.0]))

#rect = imagePyramid.pyramid[0].labelToRect()
#cv2.rectangle(imagePyramid.pyramid[0].image, rect[0], rect[1], [0, 255, 0],1 )

#[windows, labels, croppedImages] = slideWindow(imagePyramid, 4, 128)

#for i in range(0,len(labels)):
  #  print(max(labels[i]))

