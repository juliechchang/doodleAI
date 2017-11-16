# import the required libraries
import numpy as np
import time
import random
import cPickle
import codecs
import collections
import os
import math
import json
import tensorflow as tf
from six.moves import xrange

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# libraries required for visualisation:
from IPython.display import SVG, display
import svgwrite # conda install -c omnia svgwrite=1.1.6
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

# import our command line tools
from sketch_rnn_train import *
from model_mod import *
from utils import *
from rnn import *

# little function that displays vector images and saves them to .svg
def draw_strokes(data, factor=0.2, svg_filename = '/tmp/sketch_rnn/svg/sample.svg'):
  tf.gfile.MakeDirs(os.path.dirname(svg_filename))
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
  the_color = "black" #random.choice(["black","red","blue","green","orange"])
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  display(SVG(dwg.tostring()))

# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0):
  def get_start_and_end(x):
    x = np.array(x)
    x = x[:, 0:2]
    x_start = x[0]
    x_end = x.sum(axis=0)
    x = x.cumsum(axis=0)
    x_max = x.max(axis=0)
    x_min = x.min(axis=0)
    center_loc = (x_max+x_min)*0.5
    return x_start-center_loc, x_end
  x_pos = 0.0
  y_pos = 0.0
  result = [[x_pos, y_pos, 1]]

  #for each reconstruction
  for sample in s_list:
    s = sample[0]
    grid_loc = sample[1]
    grid_y = grid_loc[0]*grid_space+grid_space*0.5
    grid_x = grid_loc[1]*grid_space_x+grid_space_x*0.5
    start_loc, delta_pos = get_start_and_end(s)

    loc_x = start_loc[0]
    loc_y = start_loc[1]
    new_x_pos = grid_x+loc_x
    new_y_pos = grid_y+loc_y
    result.append([new_x_pos-x_pos, new_y_pos-y_pos, 0])

    result += s.tolist()
    result[-1][2] = 1
    x_pos = new_x_pos+delta_pos[0]
    y_pos = new_y_pos+delta_pos[1]
  return np.array(result)

def encode(input_strokes):
  strokes = to_big_strokes(input_strokes).tolist()
  strokes.insert(0, [0, 0, 1, 0, 0])
  seq_len = [len(input_strokes)]
  draw_strokes(to_normal_strokes(np.array(strokes)))
  return sess.run(eval_model.batch_z, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]

#latent vector to stroke
def decode(sample_model_var, session_var, z_input=None, draw_mode=True, temperature=0.1, factor=0.2):
  #max_seq_len = eval_model.hps.max_seq_len / 5
  z = None
  if z_input is not None:
    z = [z_input]
  sample_strokes, m = sample(session_var, sample_model_var, seq_len=5, temperature=temperature, z=z)
  strokes = to_normal_strokes(sample_strokes)
  if draw_mode:
    draw_strokes(strokes, factor)
  return strokes

#DRAW FIRST PART OF SKETCH
model_dir_1 = 'checkpoint_path/mosquito'
[hps_model_1, eval_hps_model_1, sample_hps_model_1] = load_model(model_dir_1)
reset_graph()
model_1 = Model(hps_model_1)
eval_model_1 = Model(eval_hps_model_1, reuse=True)
sample_model_1 = Model(sample_hps_model_1, reuse=True)
sess_1 = tf.InteractiveSession()
sess_1.run(tf.global_variables_initializer())
# loads the weights from checkpoint into our model
load_checkpoint(sess_1, model_dir_1)

#get some sample strokes
results = []
sample_strokes_1, m,final_state,final_x = sample(sess_1, sample_model_1, seq_len=50, temperature=0.5, z=None) #final_state,final_x
strokes_1 = to_normal_strokes(sample_strokes_1)
sketch_1 = [draw_strokes(strokes_1, 0.2),[0, 0]]
results.append(sketch_1) # append to draw out in sequence


#ATTEMPTING TO ENCODE INITIAL SKETCH (not currently working)
#sample_strokes_1 = test_set.random_sample()
#encode generated strokes to a hidden state h
strokes_h = to_big_strokes(sketch_1).tolist()
strokes_h.insert(0, [0, 0, 1, 0, 0])
seq_len = [len(sketch_1)]
draw_strokes(to_normal_strokes(np.array(strokes_h)))
h_1 = sess_1.run(eval_model_1.batch_z, feed_dict={eval_model_1.input_data: [strokes_h], eval_model_1.sequence_lengths: seq_len})[0]
#h_1 is now a latent vector


model_dir_2 = 'checkpoint_path/cat'
[hps_model_2, eval_hps_model_2, sample_hps_model_2] = load_model(model_dir_2)
reset_graph()
model_2 = Model(hps_model_2)
eval_model_2 = Model(eval_hps_model_2, reuse=True)
sample_model_2 = Model(sample_hps_model_2, reuse=True)
sess_2 = tf.InteractiveSession()
sess_2.run(tf.global_variables_initializer())
load_checkpoint(sess_2, model_dir_2)

sample_strokes_2, m = continue_sample(strokes_so_far=[], start_state=final_state, start_x=final_x, sess=sess_2, model=sample_model_2, seq_len=50, temperature=0.5, z=h_1)
strokes_2 = to_normal_strokes(sample_strokes_2)
sketch_2 = [draw_strokes(strokes_2, 0.2),[0, 1]]
results.append(sketch_2) # append to draw out in 

stroke_grid = make_grid_svg(results)
draw_strokes(stroke_grid)

#javascript program
#update the model at each step with set of strokes so far
#the new rnn state is the lstm.foward with the new input


'''
#set up second model
model_dir_2 = 'checkpoint_path/cat'
[hps_model_2, eval_hps_model_2, sample_hps_model_2] = load_model(model_dir_2)
reset_graph()
model_2 = Model(hps_model_2)
eval_model_2 = Model(eval_hps_model_2, reuse=True)
sample_model_2 = Model(sample_hps_model_2, reuse=True)
sess_2 = tf.InteractiveSession()
sess_2.run(tf.global_variables_initializer())
load_checkpoint(sess_2, model_dir_2)

#sample_strokes_1 is a partway sketch that has not been converted to normal strokes
sample_strokes_2, m = continue_sample(strokes_so_far=sample_strokes_1, start_state=final_state, start_x=final_x, sess=sess_2, model=sample_model_2, seq_len=50, temperature=0.5, z=None)
#print(sample_strokes_2)
strokes_2 = to_normal_strokes(sample_strokes_2)


#all_strokes = np.concatenate((sample_strokes_1,sample_strokes_2),axis=0)
#print(all_strokes)
#full_sketch = [draw_strokes(all_strokes, 0.2),[0, 2]]
#results.append(full_sketch)
stroke_grid = make_grid_svg(results)
draw_strokes(stroke_grid)
'''




















