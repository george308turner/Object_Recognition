from numba import cuda
import numpy
from PIL import Image, ImageDraw
import time
import copy
from cv2 import VideoCapture

loop_start_time = time.time()
center_last_found = time.time()

def get_image(camera):
    #gets image, reapeats in loop
    return_value, image = camera.read()
    image = Image.fromarray(image)

    og_size = image.size
    new_size = (720,480)
    new_im = Image.new("RGB",new_size)
    box = tuple((n-o) // 2 for n, o in zip(new_size,og_size))
    new_im.paste(image,box)

    return new_im

#takes tuple outputs image with overlayed x for the center 
def plot_center(center,f_img_array):
    image = Image.fromarray(f_img_array)
    text_image = ImageDraw.Draw(image)
    text_image.text(center,"X", fill=(150))
    image.show()

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
@cuda.jit('void(uint8[:,:],uint16[:],int16,int8)')
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
@cuda.jit('void(uint8[:,:],uint16[:],int16,int8)')
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
@cuda.jit('void(uint8[:,:],uint16[:],int16,int8)')
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
@cuda.jit('void(uint8[:,:],uint16[:],int16,int8)')
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

def circle_line_finder(p,dist_max):
    c=len(p)
    l=[]
    for i in range(c):
        m = p[i]
        current_list = [m]
        for j in range(c):
            n = p[j]
            dist = ((m[0]-n[0])**2) + ((m[1]-n[1])**2)
            if dist < dist_max:
                current_list.append(n)
        l.append(current_list)
    max_point_count = 0
    max_element = 0
    for i in range(c):
        if len(l[i]) > max_point_count:
            max_element = i
            max_point_count = len(l[i])
    max_point = l[max_element][0]
    return max_point


#kind of a minimum radius but any circles with a diameter smaller than this could be picked up
min_radius = numpy.int16(23)

#the deviation of where the white pixel on the oppposite side of the elipsis can be
expected_range = numpy.int8(4)

white_sensitivity = numpy.uint8(155)

min_contrast = numpy.int16(5800)

#built for 720 x 480 image
camera = VideoCapture(0)

#importing and formatting image datatype
img = get_image(camera)
img = img.convert("L")
img_array = numpy.array(img,dtype=numpy.uint8)
#making the list writable
img_array.setflags(write=1)


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
#image = Image.fromarray(f_img_array)
#image.show()



threadsperblock = 476

#array for all the possible circles to be added to
#format added to the index of their height, the value included is the width
possible_circles = numpy.zeros((480), dtype = numpy.uint16)
cuda.all_sync
circle_detect_l_r[blockspergrid, threadsperblock](f_img_array,possible_circles,min_radius,expected_range)
cuda.all_sync
#searching for the ones
possible_circles_l_r = numpy.array([[0,0]],dtype = numpy.uint16)
for j in range(480):
    if possible_circles[j] != 0:
        possible_circles_l_r = numpy.append(possible_circles_l_r,[[possible_circles[j],j]],axis=0)

#array for all the possible circles to be added to
#format added to the index of their height, the value included is the width
possible_circles = numpy.zeros((480), dtype = numpy.uint16)
cuda.all_sync
circle_detect_r_l[blockspergrid, threadsperblock](f_img_array,possible_circles,min_radius,expected_range)
cuda.all_sync
#searching for the ones
possible_circles_r_l = numpy.array([[0,0]],dtype = numpy.uint16)
for j in range(480):
    if possible_circles[j] != 0:
        possible_circles_r_l = numpy.append(possible_circles_r_l,[[possible_circles[j],j]],axis=0)

#array for all the possible circles to be added to
#format added to the index of their width, the value included is the height
possible_circles = numpy.zeros((720), dtype = numpy.uint16)
cuda.all_sync
circle_detect_t_b[blockspergrid, threadsperblock](f_img_array,possible_circles,min_radius,expected_range)
cuda.all_sync
#searching for the ones
possible_circles_t_b = numpy.array([[0,0]],dtype = numpy.uint16)
for i in range(720):
    if possible_circles[i] != 0:
        possible_circles_t_b = numpy.append(possible_circles_t_b,[[i,possible_circles[i]]],axis=0)

#array for all the possible circles to be added to
#format added to the index of their width, the value included is the height
possible_circles = numpy.zeros((720), dtype = numpy.uint16)
cuda.all_sync
circle_detect_b_t[blockspergrid, threadsperblock](f_img_array,possible_circles,min_radius,expected_range)
cuda.all_sync
#searching for the ones
possible_circles_b_t = numpy.array([[0,0]],dtype = numpy.uint16)
for i in range(720):
    if possible_circles[i] != 0:
        possible_circles_b_t = numpy.append(possible_circles_b_t,[[i,possible_circles[i]]],axis=0)

#finding groups of points

#fix for when top == 0 or any other coordinate
dist_max = 30
point_l = circle_line_finder(possible_circles_l_r,dist_max)
point_r = circle_line_finder(possible_circles_r_l,dist_max)
point_t = circle_line_finder(possible_circles_t_b,dist_max)
point_b = circle_line_finder(possible_circles_b_t,dist_max)


failed_list = [0,0,0,0]

if point_l[0] == 0:
    print("Left point failed")
    failed = True
    failed_list[0] = 1
if point_r[0] == 0:
    print("Right point failed")
    failed = True
    failed_list[1] = 1
if point_t[0] == 0:
    print("Top point failed")
    failed = True
    failed_list[2] = 1
if point_b[0] == 0:
    print("Bottum point failed")
    failed = True
    failed_list[3] = 1

failed = False
#center calculations
if failed_list == [0,0,0,0]:
    center_x = (point_l[0]+point_r[0]+point_t[0]+point_b[0])//4
    center_y = (point_l[1]+point_r[1]+point_t[1]+point_b[1])//4
#if either top or bottum failed
elif failed_list == [0,0,1,0] or failed_list == [0,0,0,1]:
    center_y = (point_l[1]+point_r[1])//2
    #either top or bottum will be 0
    center_x = (point_r[0]+point_l[0]+point_t[0]+point_b[0])//3
    
#if either left or right failed
elif failed_list == [1,0,0,0] or failed_list == [0,1,0,0]:
    center_x = (point_t[0] + point_b[0])//2
    center_y = (point_t[1]+point_b[1]+point_l[1]+point_r[1])//3

#top and bottum failed
elif failed_list == [0,0,1,1]:
    center_x = (point_l[0]+point_r[0])//2
    center_y = (point_l[1]+point_r[1])//2

#left and right failed
elif failed_list == [1,1,0,0]:
    center_x = (point_t[0]+point_b[0])//2
    center_y = (point_t[1]+point_b[1])//2

#top or bottum and left or right
elif failed_list == [1,0,1,0] or failed_list  == [1,0,0,1] or failed_list == [0,1,1,0] or failed_list == [0,1,0,1]:
    #either of these will be 0 so sum will be possition
    center_x = point_t + point_b
    center_y = point_l + point_r
else:
    #for during loop to know when the mug was last detected
    failed = True

if not(failed):
    center = (center_x,center_y)
    center_last_found = time.time()
    plot_center(center,f_img_array)
else:
    print("failed",str(center_last_found))




end_time = time.time()
print(end_time - s_time)