# Julie Chang, Cynthia Hua

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as random

from data_util import *

data_file = 'data/lobster.bin' # single category

count = 0
drawings = []
n_examples = 5;
for drawing in unpack_drawings(data_file):
    # do something with the drawing
    drawings.append(drawing)
    count = count + 1
    if count > n_examples - 1:
        break

# plot multiple examples
fig = plt.subplots(nrows=1, ncols=n_examples, figsize=(8,2))
for i in range(n_examples):
	image = drawings[i]['image'] # ((x,y), (x,y), ... for all strokes)
	n_strokes = len(image)
	plt.subplot(1,n_examples,i+1)
	plt.gca().invert_yaxis()
	plt.axis("off")
	plt.axis("equal")
	plt.title(i)
	for j in range(n_strokes):
		plt.plot(image[j][0], image[j][1])	

plt.show()
