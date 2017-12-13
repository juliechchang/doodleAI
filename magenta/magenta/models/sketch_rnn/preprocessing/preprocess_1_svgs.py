# import the required libraries
import numpy as np
import time
import random
import cPickle

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)
# import our command line tools

# libraries required for visualisation:
from IPython.display import SVG, display
import svgwrite
# conda install -c omnia svgwrite=1.1.6 if you don't have this lib
# helper function for draw_strokes
def get_bounds(data, factor):
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0
    
  abs_x = 0
  abs_y = 0
  for i in xrange(len(data)):
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)
    
  return (min_x, max_x, min_y, max_y)

# little function that displays vector images and saves them to .svg
def draw_strokes_mod(data, factor=0.2, svg_filename = 'sample.svg'):
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x 
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in xrange(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  return SVG(dwg.tostring())

#If the file is a .npy file, then a single array is returned.
#If the file is a .npz file, then a dictionary-like object is returned, containing {filename: array} key-value pairs, one for each file in the archive.
#npy_file = np.load("cat.npy")
#npz_file = np.load()

npz_data = np.load("cat.npz")
train_set = npz_data['train']
valid_set = npz_data['valid']
test_set = npz_data['test']

#single_img = random.choice(train_set)

for i in range(0,1000):
  current_img = train_set[i]
  svg_img = draw_strokes_mod(current_img,svg_filename="svgs/img"+str(i)+".svg")

  
# draw a random example (see draw_strokes.py)
#draw_strokes(single_img)
#cairosvg.svg2png(url="sample.svg", write_to='image.png')
