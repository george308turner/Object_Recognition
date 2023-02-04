from cv2 import VideoCapture
from PIL import Image, ImageDraw

#connect to camera only needed once in the code
camera = VideoCapture(0)


def get_image(camera):
    #gets image, reapeats in loop
    return_value, image = camera.read()
    image = Image.fromarray(image)
    image.show()
    og_size = image.size
    new_size = (720,480)
    new_im = Image.new("RGB",new_size)
    box = tuple((n-o) // 2 for n, o in zip(new_size,og_size))
    new_im.paste(image,box)

    return new_im

image = get_image(camera)

def plot_center(center):
    text_image = ImageDraw.Draw(image)
    text_image.text(center,"X", fill=(255,0,0))
    image.show()

plot_center((300,300))