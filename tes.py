from numba import cuda
import numpy
from PIL import Image
import time
import copy

#built for 720 x 480 image

#importing and formatting image datatype
img = Image.open("mugs/IMG_3919.jpg")
img = img.convert("L")


img_array = numpy.array(img,dtype=numpy.int16)
img_array.setflags(write=1)

@cuda.jit
def colour_filtering(img_array):
    width = cuda.grid(1)
    # if pos is greater than image width
    if width < 720:
        for height in range(480):
            if img_array[height][width] > 200:
                img_array[height][width] = 255
            else:
                img_array[height][width] = 0
            

@cuda.jit('void(int16[:,:],int32[:,:],int16[:,:])')
def line_detect(img_array,weighting,f_img_array):
    width = cuda.grid(1)
    if width < 716:
        for height in range(1,478):
            sum = numpy.int32(0)
            #weighting width and height
            for w_width in range(5):
                for w_height in range(5):    
                    sum += weighting[w_height][w_width] * img_array[height + w_height - 2][width + w_width]
            if sum > 5000 or sum < -5000:
                f_img_array[height][width+2] = 254
            elif f_img_array[height][width+2] == 254:
                pass
            else:
                f_img_array[height][width+2] = 0
        

threadsperblock = img.width
blockspergrid = 1

s_time = time.time()

cuda.all_sync
colour_filtering[blockspergrid, threadsperblock](img_array)
#end of black and white converstion


f_img_array = copy.deepcopy(img_array)

#vertical line detection
weighting = numpy.array([[1,2,0,-2,-1],[2,4,0,-4,-2],[3,12,0,-12,3],[2,4,0,-4,-2],[1,2,0,-2,-1]],dtype=numpy.int32)
cuda.all_sync
line_detect[blockspergrid, threadsperblock](img_array,weighting,f_img_array)
cuda.all_sync

#horizontal line detection
weighting = numpy.array([[1,2,3,2,1],[2,4,12,4,2],[0,0,0,0,0],[-2,-4,-12,-4,-2],[-1,-2,-3,-2,-1]],dtype=numpy.int32)
cuda.all_sync
line_detect[blockspergrid, threadsperblock](img_array,weighting,f_img_array)
cuda.all_sync

end_time = time.time()
print(end_time - s_time)

image = Image.fromarray(f_img_array)
image.show()


     
def circle_detect_l_to_r(f_img_array,possible_circles):

    min_radius = 35
    
    for height in range(2,478):
        for width in range(2,718-min_radius):
            if f_img_array[height][width] == 254:
                b = False
                checking_width = width + min_radius
                for checking_up_height in range(min_radius,(478-height)):
                    if f_img_array[height+checking_up_height][checking_width] == 254 and b == False:
                        if height + checking_up_height < 478:
                            for checking_down_height in range(-2,2):
                                if f_img_array[(height-checking_up_height)+checking_down_height][checking_width] == 254 and b == False:
                                    if width < 410:
                                        possible_circles = numpy.append(possible_circles,[[width,height]])
                                        b = True
    return possible_circles


possible_circles = numpy.array([],dtype=numpy.int16)

possible_circles = circle_detect_l_to_r(f_img_array,possible_circles)

print(possible_circles)
