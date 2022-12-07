from rembg import remove
from PIL import Image
import easygui as eg

input = Image.open('dumbass.jpeg')
output = remove(input)
output.save('background_less.png')
