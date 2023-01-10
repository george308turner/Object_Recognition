from PIL import Image
import time

global path
path = "720pcar.jpg"

def colour_filtering():
    #extracting pixel info
    im = Image.open(path, 'r')
    im = im.convert("L")
    width, height = im.size
    #extracting pixel map
    pixel_map = im.load()
    im.show()
    start_t = time.time()
    for i in range(width):
        for j in range(height):
            if pixel_map[i,j] > 145 and pixel_map[i,j] < 256:
                pixel_map[i,j] = 255
            else:
                pixel_map[i,j] = 0
    end_t = time.time()
    print(end_t-start_t)
    im.show()
    return pixel_map

def line_detection(pixel_map, weighting):
    im_a = Image.open(path, 'r')
    im_a = im_a.convert("L")
    pixel_map_a = im_a.load()
    width, height = im_a.size
    for i in range(width):
        #image horizontal
        for j in range(height):
            #making sure the programming isnt getting pixels off the image
            if i > (5/2) and j > (5/2) and i < width-(5/2) and j < height-(5/2):
                pix_sum = 0
                #vector vertical
                for w_j in range(-2,3):
                    #vector horizontal
                    for w_i in range(-2,3):
                        pix_sum += (weighting[w_j+2][w_i+2]*pixel_map[(i+w_i),(j+w_j)])
                if pix_sum < -40 or pix_sum > 40:
                    pixel_map_a[i,j] = 255

                else:
                    pixel_map_a[i,j] = 0
    #images all
    return im_a

def line_detection_loop():
    pixel_map = colour_filtering()
    #vertical
    weighting = [[0.004166667,0.014583333,0,-0.014583333,-0.004166667],[0.008333333,0.041666667,0,-0.041666667,-0.008333333],[0.0125,0.35,0,-0.35,-0.0125],[0.008333333,0.041666667,0,-0.041666667,-0.008333333],[0.004166667,0.014583333,0,-0.014583333,-0.004166667]]
    vertical = line_detection(pixel_map, weighting)
    print("Vertical complete")
    #horizontal
    weighting = [[0.004166667,0.008333333,0.0125,0.008333333,0.004166667],[0.014583333,0.041666667,0.35,0.041666667,0.014583333],[0,0,0,0,0],[-0.014583333,-0.041666667,-0.35,-0.041666667,-0.014583333],[-0.004166667,-0.008333333,-0.0125,-0.008333333,-0.004166667]]
    horizontal = line_detection(pixel_map, weighting)
    print("Horizontal complete")
    #diagonal top left --> bottum right
    weighting = [[0,-0.05,-0.03,-0.02,-0.01],[0.05,0,-0.08,-0.13,-0.02],[0.03,0.08,0,-0.08,-0.03],[0.02,0.13,0.08,0,-0.053],[0.01,0.02,0.03,0.05,0]]
    diagonal_tl_br = line_detection(pixel_map, weighting)
    print("Diangonal_1 complete")
    #diagonal top right --> bottum left
    weighting = [[0.01,0.02,0.03,0.05,0],[0.02,0.13,0.08,0,-0.05],[0.03,0.08,0,-0.08,-0.03],[0.05,0,-0.08,-0.13,-0.02],[0,-0.05,-0.03,-0.02,-0.01]]
    diagonal_tr_bl = line_detection(pixel_map, weighting)
    print("Diangonal_2 complete")
    return vertical, horizontal, diagonal_tl_br, diagonal_tr_bl

def line_combine():
    vertical, horizontal, diagonal_tl_br, diagonal_tr_bl = line_detection_loop()
    #extracting pixel info
    final = Image.open(path, 'r')
    final = final.convert("1")
    width, height = final.size
    #extracting pixel map
    pixel_map = final.load()

    pixel_map_v = vertical.load()
    pixel_map_h = horizontal.load()
    pixel_map_tl_br = diagonal_tl_br.load()
    pixel_map_tr_bl = diagonal_tr_bl.load()
    for i in range(width):
        for j in range(height):
            try:
                pixel_map[i,j] = max(pixel_map_v[i,j],pixel_map_h[i,j],pixel_map_tl_br[i,j],pixel_map_tr_bl[i,j])
            except:
                print(i,j)
    final.show()
    return final

def line_finder():
    image = line_combine()

s_time = time.time()
line_finder()
f_time = time.time()

print(f_time-s_time)