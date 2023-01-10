import numpy


img_array = numpy.array([1,2,3,4])
img_array.setflags(write = 1)
img_array[2] = 7
print(img_array)
