import numpy as np
import cv2
import math
from PIL import Image
import matplotlib.pyplot as plt

paths = []
ellipses = []
with open('annotations_short.txt') as inputfile:
    prevLine = 0
    getNextLine = 0
    for line in inputfile:
        line = line.strip()
        if(len(line)==1 and int(line)==1):
            paths.append(prevLine)
            getNextLine = 1
        elif(getNextLine):
            ellipses.append(line)
            getNextLine = 0
        prevLine = line

#print(paths)
#print(ellipses)


imdb = []
for path in paths:
    pil_im = Image.open('images/' + path + '.jpg').convert('L')
    img = np.array(pil_im)
    img = img - np.mean(img)
    img = img/np.std(img)
    imdb.append(img)


#for im in imdb:
    #plt.imshow(im,cmap=plt.cm.gray)
    #plt.show()

rectangles = []

for ellipse in ellipses:
	ellipse = ellipse.split(' ')
	rmax = float(ellipse[0])
	rmin = float(ellipse[1])
	angle = float(ellipse[2])
	x_center = float(ellipse[3])
	y_center = float(ellipse[4])
	width = rmax*math.sin(angle)
	rectangles.append([x_center, y_center, width])
	#cv2.rectangles()

print(rectangles)



