from rembg import remove
from PIL import Image
import os

import tensorflow as tf

input_path = 'test.jpeg'
input = Image.open(input_path)
output = remove(input)
output.save('test' + '.png')

loaded_png = tf.io.read_file('test.png')
img = tf.image.decode_png(loaded_png)
data = tf.image.resize(img, (224, 224))
data = data/255.0

print(data)


