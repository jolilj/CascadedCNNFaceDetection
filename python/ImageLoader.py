import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

paths = []
ellipses = []
with open('annotations.txt') as inputfile:
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

print(paths)
print(ellipses)


imdb = []
for path in paths:
    pil_im = Image.open(path + '.jpg').convert('L')
    img = np.array(pil_im)
    img = img - np.mean(img)
    img = img/np.std(img)
    imdb.append(img)


for im in imdb:
    plt.imshow(im,cmap=plt.cm.gray)
    plt.show()
