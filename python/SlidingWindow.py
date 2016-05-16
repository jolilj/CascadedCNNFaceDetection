import numpy as np
import cv2
import time
import math
from imagepyramid import ImagePyramid
from PIL import Image
import matplotlib.pyplot as plt


# So far this is just a nice script without any purpose in life. We should probably turn it into a function. Then it can take the image, label, step size, winLength as arguments and become useful. But now I am going home. If I manage to use this git hub thing. Goodbye python :)

imdb = []

    
pil_im = Image.open('/home/c/a/carhei/FaceDetectionRaspberryPi/python/test_im.jpg').convert('L')
img = np.array(pil_im)
img = img - np.mean(img)
img = img/np.std(img)
imdb.append(img)

#taking windows of the specified size from all images in imdb 


for i in range(0, len(imdb)):

	image_width = imdb[i].shape[1]
	image_height = imdb[i].shape[0]

	stepSize = int(image_width/30)
	windowLength = int(image_width/5)
	

	for y in np.arange(0,image_height-windowLength,stepSize):
		for x in np.arange(0,image_width-windowLength,stepSize):
			image = imdb[i]
			window = [x,y, image[x:x+windowLength], image[y:y+windowLength]]
			copy = image.copy()
			cv2.rectangle(copy, (x,y), (x+windowLength, y+windowLength), [255, 255, 255],1 )
			cv2.imshow("Window", copy)
			cv2.waitKey(1)
			time.sleep(0.03)



