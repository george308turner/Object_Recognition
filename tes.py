from numba import cuda
import numpy
from PIL import Image
import time


img = Image.open("720pcar.jpg")
img = img.convert("L")

img_array = numpy.array(img,dtype=numpy.int16)
img_array.setflags(write=1)

weighting = numpy.array([[1,2,0,-2,-1],[2,4,0,-4,-2],[3,12,0,-12,3],[2,4,0,-4,-2],[1,2,-0,-2,-1]],dtype=numpy.int16)

width=0

while True:
    
    if width < 716:
        for height in range(1,478):
            sum = numpy.int16("0")
            #weighting width and height
            for w_width in range(5):
                for w_height in range(5):
                    sum += img_array[height + w_height - 2][width + w_width] * weighting[w_height][w_width]
            
            if sum > 10000:
                img_array[height][width+2] = 255
            else:
                img_array[height][width+2] = 0
        width+=1
    else:
        break

image = Image.fromarray(img_array)
image.show()