from numba import cuda
import numpy
from PIL import Image
import time
import copy

#best values so far
#23, 4, 185, 7000

#kind of a minimum radius but any circles with a diameter smaller than this could be picked up
min_radius = numpy.int16(23)

#the deviation of where the white pixel on the oppposite side of the elipsis can be
expected_range = numpy.int8(4)

white_sensitivity = numpy.uint8(185)

min_contrast = numpy.int16(7000)

#built for 720 x 480 image

#importing and formatting image datatype
img = Image.open("mugs/IMG_3917.jpg")
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
@cuda.jit('void(uint8[:,:],uint8[:],int16,int8)')
def circle_detect_l_r(f_img_array,possible_circles,min_radius,expected_range):
    height = cuda.grid(1)
    if height < 479:
        for width in range(2,718-min_radius):
            if f_img_array[height][width] == numpy.uint8(254):
                checking_width = width + min_radius
                for checking_up_height in range(5,(478-height)):
                    if f_img_array[height+checking_up_height][checking_width] == numpy.uint8(254):
                        if height + checking_up_height < 478:
                            for checking_down_height in range(-(expected_range),expected_range):
                                if f_img_array[(height-checking_up_height)+checking_down_height][checking_width] == numpy.uint8(254):
                                    if width < 410:
                                        possible_circles[height] = width
                                        break

#right to left
@cuda.jit('void(uint8[:,:],uint8[:],int16,int8)')
def circle_detect_r_l(f_img_array,possible_circles,min_radius,expected_range):
    height = cuda.grid(1)
    if height < 479:
        for width in range(2+min_radius,718):
            if f_img_array[height][width] == numpy.uint8(254):
                checking_width = width - min_radius
                for checking_up_height in range(5,(478-height)):
                    if f_img_array[height+checking_up_height][checking_width] == numpy.uint8(254):
                        if height + checking_up_height < 478:
                            for checking_down_height in range(-(expected_range),expected_range):
                                if f_img_array[(height-checking_up_height)+checking_down_height][checking_width] == numpy.uint8(254):
                                    if width < 410:
                                        possible_circles[height] = width
                                        break

#top to bottum
@cuda.jit('void(uint8[:,:],uint8[:],int16,int8)')
def circle_detect_t_b(f_img_array,possible_circles,min_radius,expected_range):
    width = cuda.grid(1)
    if width < 719:
        for height in range(2,478-min_radius):
            if f_img_array[height][width] == numpy.uint8(254):
                checking_height = height + min_radius
                for checking_left_width in range(5,(718-width)):
                    if f_img_array[checking_height][width+checking_left_width] == numpy.uint8(254):
                        if width + checking_left_width < 718:
                            for checking_right_width in range(-(expected_range),expected_range):
                                if f_img_array[checking_height][(width-checking_left_width)+checking_right_width] == numpy.uint8(254):
                                    if height < 770:
                                        possible_circles[width] = height

#bottum to top
@cuda.jit('void(uint8[:,:],uint8[:],int16,int8)')
def circle_detect_b_t(f_img_array,possible_circles,min_radius,expected_range):
    width = cuda.grid(1)
    if width < 719:
        for height in range(2+min_radius,478):
            if f_img_array[height][width] == numpy.uint8(254):
                checking_height = height - min_radius
                for checking_left_width in range(5,(718-width)):
                    if f_img_array[checking_height][width+checking_left_width] == numpy.uint8(254):
                        if width + checking_left_width < 718:
                            for checking_right_width in range(-(expected_range),expected_range):
                                if f_img_array[checking_height][(width-checking_left_width)+checking_right_width] == numpy.uint8(254):
                                    if height < 770:
                                        possible_circles[width] = height

def reject_outliers_average(data, m=100):
    sum_i = 0
    sum_j = 0
    length = len(data)
    for index in range(length):
        sum_i += data[index][0]
        sum_j += data[index][1]
    mean_i = sum_i // length
    mean_j = sum_j // length

    num_rejected = 0
    max_i = mean_i + m
    min_i = mean_i - m
    max_j = mean_j + m
    min_j = mean_j - m
    for index in range(length):
        if data[index-num_rejected][0] < min_i or data[index-num_rejected][0] > max_i or data[index-num_rejected][1] < min_j or data[index-num_rejected][1] > max_j:
            data = numpy.delete(data,index-num_rejected,0)
            num_rejected += 1

    length = len(data)
    for index in range(length):
        sum_i += data[index][0]
        sum_j += data[index][1]
    mean_i = sum_i // length
    mean_j = sum_j // length

    return (mean_i,mean_j)

#Matching lines horizontal
#@cuda.jit('void()')
#def matching_lines_h()

s_time = time.time()

threadsperblock = img.width
blockspergrid = 1

cuda.all_sync
colour_filtering[blockspergrid, threadsperblock](img_array, white_sensitivity)
#end of black and white converstion

#Line Detection
###############
f_img_array = copy.deepcopy(img_array)

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
image = Image.fromarray(f_img_array)
image.show()

threadsperblock = 476


#array for all the possible circles to be added to
#format added to the index of their height, the value included is the width
possible_circles = numpy.zeros((480), dtype = numpy.uint8)
cuda.all_sync
circle_detect_l_r[blockspergrid, threadsperblock](f_img_array,possible_circles,min_radius,expected_range)
cuda.all_sync
#searching for the ones
possible_circles_l_r = numpy.array([[0,0]],dtype = numpy.uint8)
for j in range(480):
    if possible_circles[j] != 0:
        possible_circles_l_r = numpy.append(possible_circles_l_r,[[possible_circles[j],j]],axis=0)

#array for all the possible circles to be added to
#format added to the index of their height, the value included is the width
possible_circles = numpy.zeros((480), dtype = numpy.uint8)
cuda.all_sync
circle_detect_r_l[blockspergrid, threadsperblock](f_img_array,possible_circles,min_radius,expected_range)
cuda.all_sync
#searching for the ones
possible_circles_r_l = numpy.array([[0,0]],dtype = numpy.uint8)
for j in range(480):
    if possible_circles[j] != 0:
        possible_circles_r_l = numpy.append(possible_circles_r_l,[[possible_circles[j],j]],axis=0)

#array for all the possible circles to be added to
#format added to the index of their width, the value included is the height
possible_circles = numpy.zeros((720), dtype = numpy.uint8)
cuda.all_sync
circle_detect_t_b[blockspergrid, threadsperblock](f_img_array,possible_circles,min_radius,expected_range)
cuda.all_sync
#searching for the ones
possible_circles_t_b = numpy.array([[0,0]],dtype = numpy.uint8)
for i in range(720):
    if possible_circles[i] != 0:
        possible_circles_t_b = numpy.append(possible_circles_t_b,[[i,possible_circles[i]]],axis=0)

#array for all the possible circles to be added to
#format added to the index of their width, the value included is the height
possible_circles = numpy.zeros((720), dtype = numpy.uint8)
cuda.all_sync
circle_detect_b_t[blockspergrid, threadsperblock](f_img_array,possible_circles,min_radius,expected_range)
cuda.all_sync
#searching for the ones
possible_circles_b_t = numpy.array([[0,0]],dtype = numpy.uint8)
for i in range(720):
    if possible_circles[i] != 0:
        possible_circles_b_t = numpy.append(possible_circles_b_t,[[i,possible_circles[i]]],axis=0)

#finding matching lines

#reject outlieres

#average

print(possible_circles_l_r)

point_l = reject_outliers_average(possible_circles_l_r)
point_r = reject_outliers_average(possible_circles_r_l)
point_t = reject_outliers_average(possible_circles_t_b)
point_b = reject_outliers_average(possible_circles_b_t)

i_center = (point_l[0] + point_r[0] + point_t[0] + point_b[0])//4
j_center = (point_l[1] + point_r[1] + point_t[1] + point_b[1])//4

print(i_center,j_center)

end_time = time.time()
print(end_time - s_time)