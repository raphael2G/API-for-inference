from rembg import remove
from PIL import Image
import os

import tensorflow as tf

input_path = 'background.jpeg'
input = Image.open(input_path)
output = remove(input)
output.save('background_removed.png')

background_removed=Image.open('background_removed.png')
imageBox = background_removed.getbbox()
cropped = background_removed.crop(imageBox)
cropped.save('cropped.png')

background_cropped_png = Image.open('cropped.png')
removed_transparency = background_cropped_png.convert('RGB')
background_cropped_jpg = removed_transparency.save('cropped.jpg')
loaded_png = tf.io.read_file('cropped.jpg')
img = tf.image.decode_image(loaded_png)
data = tf.image.resize(img, (224, 224))
data = data/255.0

print(data)


