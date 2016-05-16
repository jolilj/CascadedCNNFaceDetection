import numpy as np
import cv2
import math
from imagepyramid import ImagePyramid
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
ellipsedata = []

for ellipse in ellipses:
	ellipse = ellipse.split(' ')
	rmax = float(ellipse[0])
	rmin = float(ellipse[1])
	angle = float(ellipse[2])
	x_center = float(ellipse[3])
	y_center = float(ellipse[4])
	width = int(2*rmax*math.cos(angle*math.pi/180))
	ellipsedata.append([(int(x_center), int(y_center)), [int(angle)], (int(rmax), int(rmin))])
	rectangles.append([x_center, y_center, math.fabs(width)])
	#cv2.rectangles()

print(rectangles)


pyr = ImagePyramid(imdb[1], np.asarray(rectangles[1]))

rect = pyr.pyramid[1].labelToRect()
cv2.rectangle(pyr.pyramid[1].image, rect[0], rect[1], [0, 255, 0],1 )

#plt.imshow(pyr.pyramid[0].image,cmap=plt.cm.gray)
#plt.show()

print("===================")
print(ellipsedata[1][0])
print(ellipsedata[1][1][0])
print(ellipsedata[1][2])
cv2.ellipse(pyr.pyramid[1].image, ellipsedata[1][0], ellipsedata[1][2], 90+ellipsedata[1][1][0], 0, 360, [0, 255,0])

plt.imshow(pyr.pyramid[1].image,cmap=plt.cm.gray)
plt.show()
