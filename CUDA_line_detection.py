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

img_array = numpy.array(img,dtype=numpy.int16)
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
            

@cuda.jit('void(int16[:,:],int32[:,:],int16[:,:])')
def line_detect(img_array,weighting,v_img_array):
    width = cuda.grid(1)
    if width < 716:
        for height in range(1,478):
            sum = numpy.int32(0)
            #weighting width and height
            for w_width in range(5):
                for w_height in range(5):    
                    sum += weighting[w_height][w_width] * img_array[height + w_height - 2][width + w_width]
            if sum > 5000:
                v_img_array[height][width+2] = 255
            else:
                v_img_array[height][width+2] = 0
        

threadsperblock = img.width
blockspergrid = 1

s_time = time.time()

cuda.all_sync
colour_filtering[blockspergrid, threadsperblock](img_array)
#end of black and white converstion

#vertical line detection
weighting = numpy.array([[1,2,0,-2,-1],[2,4,0,-4,-2],[3,12,0,-12,3],[2,4,0,-4,-2],[1,2,0,-2,-1]],dtype=numpy.int32)
v_img_array = copy.deepcopy(img_array)
cuda.all_sync
line_detect[blockspergrid, threadsperblock](img_array,weighting,v_img_array)
cuda.all_sync

end_time = time.time()
print(end_time - s_time)

image = Image.fromarray(v_img_array)
image.show()


