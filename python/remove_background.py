from rembg import remove
from PIL import Image
import os

input_path = 'dumbass.jpeg'
input = Image.open('dumbass.jpeg')
output = remove(input)
output.save(os.path.split(input_path))
