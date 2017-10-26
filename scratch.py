# Julie Chang, Cynthia Hua

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as random
import math 

from data_util import *

data_1 = 'data/banana.bin' # keyword 1
data_2 = 'data/flower.bin' # keyword 2

count = 0
drawings_1 = []
drawings_2 = []
n_examples = 8;
for drawing in unpack_drawings(data_1):
    # do something with the drawing
    drawings_1.append(drawing)
    count = count + 1
    if count > n_examples - 1:
        break

count = 0
for drawing in unpack_drawings(data_2):
    # do something with the drawing
    drawings_2.append(drawing)
    count = count + 1
    if count > n_examples - 1:
        break


fig = plt.subplots(nrows=3, ncols=n_examples, figsize=(11,5))

# plot multiple examples
for i in range(n_examples):
	image_1 = drawings_1[i]['image'] # ((x,y), (x,y), ... for all strokes)
	n_strokes = len(image_1)
	plt.subplot(3,n_examples,i+1)
	plt.gca().invert_yaxis()
	plt.axis("off")
	plt.axis("equal")
	plt.title(i)
	# colors = [(.1, .1, 1) for j in range(n_strokes)]
	colors = [(.1 + j*.8/n_strokes, .1, 1 - j*.9/n_strokes) for j in range(n_strokes)]
	for j in range(n_strokes):
		plt.plot(image_1[j][0], image_1[j][1], color = colors[j], linewidth = 1)	

	image_2 = drawings_2[i]['image'] # ((x,y), (x,y), ... for all strokes)
	n_strokes = len(image_2)
	plt.subplot(3,n_examples,n_examples + i+1)
	plt.gca().invert_yaxis()
	plt.axis("off")
	plt.axis("equal")
	plt.title(i)
	# colors = [(.8, .1, .1) for j in range(n_strokes)]
	colors = [(.1 + j*.8/n_strokes, .1, 1 - j*.9/n_strokes) for j in range(n_strokes)]
	for j in range(n_strokes):
		plt.plot(image_2[j][0], image_2[j][1], color = colors[j], linewidth = 1)	

# try to combine 
n_comb = n_examples
for i in range(n_comb):
	# draw half of image 1
	image_1 = drawings_1[i]['image'] # ((x,y), (x,y), ... for all strokes)
	n_strokes = len(image_1)
	plt.subplot(3,n_examples,2*n_examples + i+1)
	plt.gca().invert_yaxis()
	plt.axis("off")
	plt.axis("equal")
	plt.title('comb')
	color = (.1, .1, 1)
	if n_strokes > 1:
		for j in range(round(n_strokes/2)):
			plt.plot(image_1[j][0], image_1[j][1], color = color, linewidth = 1.0)	
	else:
		n_points = len(image_1[0][0])
		half = round(n_points/2)
		xvals = image_1[0][0][:half]
		yvals = image_1[0][1][:half]
		plt.plot(xvals, yvals, color = color, linewidth = 1.0)
	
	# draw half of image 2
	image_2 = drawings_2[i]['image'] # ((x,y), (x,y), ... for all strokes)
	n_strokes = len(image_2)
	color = (.8, .1, .1)
	if n_strokes > 1:
		for j in range(math.floor(n_strokes/2),n_strokes):
			plt.plot(image_2[j][0], image_2[j][1], color = color, linewidth = 1.0)	
	else:
		n_points = len(image_2[0][0])
		half = math.floor(n_points/2)
		xvals = image_2[0][0][:half]
		yvals = image_2[0][1][:half]
		plt.plot(xvals, yvals, color = color, linewidth = 1.0)
	

# plt.show()

# save images 
plt.savefig('bananaflower.pdf', bbox_inches='tight')
