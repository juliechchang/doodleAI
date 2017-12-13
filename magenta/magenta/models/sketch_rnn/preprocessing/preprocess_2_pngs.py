# import the required libraries
import numpy as np
import time
import random
import _pickle as cPickle

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)
# import our command line tools
import cairosvg

# libraries required for visualisation:
from IPython.display import SVG, display
import svgwrite

import imageio

#svg_img = draw_strokes_mod(single_img)
for i in range(0,100):
	file = "img"+str(i)+".svg"
	new_file = "pngs/img"+str(i)+".png"
	cairosvg.svg2png(url=file, write_to=new_file)
#im = imageio.imread('image.png')
#print(im.shape)
