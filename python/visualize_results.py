from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imagepyramid
##Visualize the result in a figure
#   *pImages - the predicted images
#   *pSubImages - the predicted subimages
#   *pWindows - the corresponding window of the subimages
#   *tImages - true images
#   *tSubImages - true subimages
#   *tWindows - true windows

def visualizeResult(title,pImages, pSubImages, pWindows, tImages=[], tSubImages=[], tWindows=[]):
    for i in range(0,len(pImages)):
        pImage = pImages[i]
        pSubImage = pSubImages[i]
        pWindow = pWindows[i]

        predicted_rect = imagepyramid.labelToRect(pWindow)
        cv2.rectangle(pImage, predicted_rect[0], predicted_rect[1], [0, 255, 0],1 )

        fig = plt.figure(title)
        fig.add_subplot(2,2,1)
        plt.imshow(pImage,cmap=plt.cm.gray)
        fig.add_subplot(2,2,2)
        plt.imshow(pSubImage, cmap=plt.cm.gray)

        #if targets are available
        if (i < len(tImages)):
            tImage = tImages[i]
            tSubImage = tSubImages[i]
            tWindow = tWindows[i]

            true_rect =  imagepyramid.labelToRect(tWindow)
            cv2.rectangle(tImage, true_rect[1], true_rect[0], [0, 255, 0],1 )

            fig.add_subplot(2,2,3)
            plt.imshow(tImage,cmap=plt.cm.gray)
            fig.add_subplot(2,2,4)
            plt.imshow(tSubImage, cmap=plt.cm.gray)
        plt.show()

def visualizeResultNoSubImage(title, pImages, pWindows):
    for i in range(0,len(pImages)):
        pImage = pImages[i]
        pWindow = pWindows[i]

        predicted_rect = imagepyramid.labelToRect(pWindow)
        cv2.rectangle(pImage, predicted_rect[0], predicted_rect[1], [0, 0, 0],1 )

        plt.imshow(pImage,cmap=plt.cm.gray)
        plt.show()

