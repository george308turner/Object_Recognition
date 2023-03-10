from numba import cuda
import numpy
from PIL import Image
import time


#importing and formatting image datatype
img = Image.open("720pcar.jpg")
img = img.convert("L")

img_array = numpy.array(img)
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
            

threadsperblock = img.width

blockspergrid = 1

s_time = time.time()

cuda.all_sync
colour_filtering[blockspergrid, threadsperblock](img_array)
cuda.all_sync

end_time = time.time()

print(end_time - s_time)

image = Image.fromarray(img_array)
image.show()