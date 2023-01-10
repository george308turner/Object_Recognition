from numba import cuda
import numpy
import math

print(cuda.gpus)

@cuda.jit
def my_func(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2

data = numpy.ones(256)

threadsperblock = 256
blockspergrid = math.ceil(data.shape[0]/threadsperblock)
print(blockspergrid)
my_func[blockspergrid, threadsperblock](data)

print(data)
print(data.shape[0])


