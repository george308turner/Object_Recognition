from numba import cuda
import numpy
from PIL import Image
import time
import copy
import sys

#built for 720 x 480 image

#importing and formatting image datatype
img = Image.open("720pcar.jpg")
img = img.convert("L")

img_array = numpy.array(img,dtype=numpy.uint8)
img_array.setflags(write=1)

@cuda.jit
def colour_filtering(img_array):
    width = cuda.grid(1)
    # if pos is greater than image width
    if width < 720:
        for height in range(480):
            if img_array[height][width] > 145:
                img_array[height][width] = 255
            else:
                img_array[height][width] = 0
            
@cuda.jit#('void(numpy.int32[:],numpy.int32[:],numpy.float64[:])')
def line_detect(img_array,weighting): #og_img_array,img_array,weighting
    width = cuda.grid(1)
    if width < 720:
        for height in range(480):
            sum = 0
            #weighting width and height
            for w_width in range(5):
                for w_height in range(5):
                    sum += img_array[height + w_height - 2][width + w_width -2] * weighting[w_height][w_width]
            cuda.all_sync
            if (sum*240) > 145:
                img_array[width][height] = 255
            else:
                img_array[width][height] = 0






threadsperblock = img.width
blockspergrid = 1



s_time = time.time()

cuda.all_sync
colour_filtering[blockspergrid, threadsperblock](img_array)
cuda.all_sync


#end of black and white converstion



#vertical line detection
weighting = numpy.array([[1,2,0,-2,-1],[2,4,0,-4,-2],[3,12,0,-12,3],[2,4,0,-4,-2],[1,2,-0,-2,-1]],dtype=numpy.uint8)


threadsperblock = img.width
blockspergrid = 1

cuda.all_sync
line_detect[blockspergrid, threadsperblock](img_array,weighting)
cuda.all_sync
#img_array = img_array.copy_to_host()



end_time = time.time()
print(end_time - s_time)

image = Image.fromarray(img_array)
image.show()


