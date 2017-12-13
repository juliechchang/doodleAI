# Imports
import tensorflow as tf
import numpy as np
import pandas as pd
import rnn

# define the CNN model
def cnn_model(features, labels=None, mode=None):
  """Model function for CNN."""

  # Input Layer [batch_size, image_width, image_height, channels]
  # we have 28 by 28 .npy file type images that we want to reshape to ^  
  # -1 for batch size allows us to treat batch size a hyperparameter we can tune
  input_layer = tf.reshape(features, [-1, 48, 48, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=4,
      kernel_size=[2, 2],
      strides=(2,2),
      padding="same",
      activation=tf.nn.relu)
  # output: [batch_size, 24, 24, 4]

  # Convolutional Layer #2 
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=4,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
  # output: [batch_size, 24, 24, 4]

  # Convolutional Layer #3
  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=8,
      strides=(2,2),
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
  # output: [batch_size, 12, 12, 8]
    
  # Convolutional Layer #4
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=8,
      strides=(2,2),
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
  # output: [batch_size, 6, 6, 8]

  # Convolutional Layer #5
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=8,
      strides=(1,1),
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
  # output: [batch_size, 6, 6, 8]

  # flatten our feature map to shape [batch_size, features], so that our tensor has only two dimensions:
  y_out = tf.reshape(conv5, [-1, 6 * 6 * 8])
  
  # return final layer
  return y_out

def super_linear(x,
                 output_size,
                 scope=None,
                 reuse=False,
                 init_w='gaussian',
                 weight_start=0.001,
                 use_bias=True,
                 bias_start=0.0,
                 input_size=None):
  """Performs linear operation. Uses ortho init defined earlier."""
  shape = x.get_shape().as_list()
  with tf.variable_scope(scope or 'linear'):
    if reuse is True:
      tf.get_variable_scope().reuse_variables()

    w_init = None  # uniform
    if input_size is None:
      x_size = shape[1]
    else:
      x_size = input_size
    if init_w == 'zeros':
      w_init = tf.constant_initializer(0.0)
    elif init_w == 'constant':
      w_init = tf.constant_initializer(weight_start)
    elif init_w == 'gaussian':
      w_init = tf.random_normal_initializer(stddev=weight_start)

    w = tf.get_variable(
        'super_linear_w', [x_size, output_size], tf.float32, initializer=w_init)
    if use_bias:
      b = tf.get_variable(
          'super_linear_b', [output_size],
          tf.float32,
          initializer=tf.constant_initializer(bias_start))
      return tf.matmul(x, w) + b
    return tf.matmul(x, w)

# #LOAD DATA FOR CNN
# cat = np.load('cat.npy')
# mosquito = np.load('mosquito.npy')

# # add a column with labels, 0=cat
# cat = np.c_[cat, np.zeros(len(cat))]
# mosquito = np.c_[mosquito, np.ones(len(mosquito))]

# # merge the arrays, and split the features (X) and labels (y). Convert to float32 to save some memory.
# features_data = np.concatenate((cat[:5000,:-1], mosquito[:5000,:-1]), axis=0).astype('float32') # all columns but the last
# labels_data = np.concatenate((cat[:5000,-1], mosquito[:5000,-1]), axis=0).astype('float32') # the last column

# # train/test split (divide by 255 to obtain normalized values between 0 and 1)
# features_train, features_test, labels_train, labels_test = train_test_split(X/255.,y,test_size=0.5,random_state=0)

# # one hot encode outputs
# labels_train_cnn = np_utils.to_categorical(labels_train)
# labels_test_cnn = np_utils.to_categorical(labels_test)
# num_classes = labels_test_cnn.shape[1]
# # reshape to be [samples][pixels][width][height]
# features_train_cnn = features_train.reshape(features_train.shape[0], 1, 28, 28).astype('float32')
# features_test_cnn = features_test.reshape(features_test.shape[0], 1, 28, 28).astype('float32')

