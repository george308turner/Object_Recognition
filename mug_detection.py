from numba import cuda
import numpy
from PIL import Image
import time
import copy

#built for 720 x 480 image

#importing and formatting image datatype
img = Image.open("mugs/IMG_3913.jpg")
img = img.convert("L")
img_array = numpy.array(img,dtype=numpy.uint8)
#making the list writable
img_array.setflags(write=1)

@cuda.jit('void(uint8[:,:],uint8)')
def colour_filtering(img_array,white_sensitivity):
    width = cuda.grid(1)
    # if pos is greater than image width
    if width < 720:
        for height in range(480):
            if img_array[height][width] > white_sensitivity:
                img_array[height][width] = numpy.uint8(255)
            else:
                img_array[height][width] = numpy.uint8(0)
            
@cuda.jit('void(uint8[:,:],int32[:,:],uint8[:,:],int16)')
def line_detect(img_array,weighting,f_img_array,min_contrast):
    width = cuda.grid(1)
    if width < 716:
        for height in range(1,478):
            sum = numpy.int32(0)
            #weighting width and height
            for w_width in range(5):
                for w_height in range(5):    
                    sum += weighting[w_height][w_width] * img_array[height + w_height - 2][width + w_width]
            if sum > min_contrast or sum < -(min_contrast):
                f_img_array[height][width+2] = numpy.uint8(254)
            elif f_img_array[height][width+2] == numpy.uint8(254):
                pass
            else:
                f_img_array[height][width+2] = numpy.uint8(0)
        
#left to right
@cuda.jit('void(uint8[:,:],uint8[:],uint8,int8)')
def circle_detect_l_r(f_img_array,possible_circles,min_radius,expected_range):
    height = cuda.grid(1)
    if height < 479:
        for width in range(2,718-min_radius):
            if f_img_array[height][width] == 254:
                checking_width = width + min_radius
                for checking_up_height in range(5,(478-height)):
                    if f_img_array[height+checking_up_height][checking_width] == 254:
                        if height + checking_up_height < 478:
                            for checking_down_height in range(-(expected_range),expected_range):
                                if f_img_array[(height-checking_up_height)+checking_down_height][checking_width] == 254:
                                    if width < 410:
                                        possible_circles[height] = width
                                        break


s_time = time.time()

threadsperblock = img.width
blockspergrid = 1

white_sensitivity = numpy.uint8(200)
cuda.all_sync
colour_filtering[blockspergrid, threadsperblock](img_array, white_sensitivity)
#end of black and white converstion


f_img_array = copy.deepcopy(img_array)
min_contrast = numpy.int16(5000)
#vertical line detection
weighting = numpy.array([[1,2,0,-2,-1],[2,4,0,-4,-2],[3,12,0,-12,3],[2,4,0,-4,-2],[1,2,0,-2,-1]],dtype=numpy.int32)
cuda.all_sync
line_detect[blockspergrid, threadsperblock](img_array,weighting,f_img_array,min_contrast)
cuda.all_sync

#horizontal line detection
weighting = numpy.array([[1,2,3,2,1],[2,4,12,4,2],[0,0,0,0,0],[-2,-4,-12,-4,-2],[-1,-2,-3,-2,-1]],dtype=numpy.int32)
cuda.all_sync
line_detect[blockspergrid, threadsperblock](img_array,weighting,f_img_array,min_contrast)
cuda.all_sync



#only required for image output
#image = Image.fromarray(f_img_array)
#image.show()



threadsperblock = 476

#array for all the possible circles to be added to
#format added to the index of their height, the value included is the width
possible_circles = numpy.zeros((480), dtype = numpy.uint8)

#kind of a minimum radius but any circles with a diameter smaller than this could be picked up
min_radius = numpy.uint8(20)

#the deviation of where the white pixel on the oppposite side of the elipsis can be
expected_range = numpy.int8(5)

cuda.all_sync
circle_detect_l_r[blockspergrid, threadsperblock](f_img_array,possible_circles,min_radius,expected_range)
cuda.all_sync

#searching for the ones
possible_circles_l_r = []
for j in range(480):
    if possible_circles[j] != 0:
        possible_circles_l_r.append([possible_circles[j],j])


#finding the median point
print(possible_circles_l_r)
circle_l = possible_circles_l_r[len(possible_circles_l_r)//2]

print(circle_l)





radius = 0

for i in range((circle_l[0]+5),720):
    if f_img_array[circle_l[1]][i] == 254:
        radius = i - circle_l[0] 
        break

print(radius)

end_time = time.time()
print(end_time - s_time)