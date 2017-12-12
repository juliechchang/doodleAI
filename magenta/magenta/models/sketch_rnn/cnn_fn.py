# Imports
import tensorflow as tf
import numpy as np
import pandas as pd
import rnn

# define the CNN model
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""

  # Input Layer [batch_size, image_width, image_height, channels], #we have 28 by 28 .npy file type images that we want to reshape to ^  
  #-1 for batch size allows us to treat batch size a hyperparameter we can tune
  input_layer = tf.reshape(features, [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5], #aka kernel_size=5
      padding="same", #padding=same here, which instructs TensorFlow to add 0 values to the edges of the output tensor to preserve width and height of 28
      activation=tf.nn.relu)
  #output: [batch_size, 28, 28, 32]: the same width and height dimensions as the input, but now with 32 channels holding the output from each of the filters.

  # Pooling Layer #1 - construct a layer that performs max pooling with a 2x2 filter and stride of 2:
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  #convolutional layer #2 takes the output tensor of our first pooling layer (pool1) as input, and produces the tensor h_conv2 as output. conv2 has a shape of [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  #flatten our feature map (pool2) to shape [batch_size, features], so that our tensor has only two dimensions:
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  
  #Dense Layer: Next, we want to add a dense layer (with 1,024 neurons and ReLU activation) to our CNN to perform classification on the features extracted by the convolution/pooling layers.
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  
  #Dropout: To help improve the results of our model, we also apply dropout regularization to our dense layer, using the dropout method in layers:
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.8, training=mode == learn.ModeKeys.TRAIN)
  #Our output tensor dropout has shape [batch_size, 1024].
  '''
  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  '''
  final_layer = logits
  return final_layer
  #print(sess.run(tf.argmax(final_layer, 1), feed_dict={x: mnist.test.images}))

#LOAD DATA FOR CNN
cat = np.load('cat.npy')
mosquito = np.load('mosquito.npy')

# add a column with labels, 0=cat
cat = np.c_[cat, np.zeros(len(cat))]
mosquito = np.c_[mosquito, np.ones(len(mosquito))]

# merge the arrays, and split the features (X) and labels (y). Convert to float32 to save some memory.
features_data = np.concatenate((cat[:5000,:-1], mosquito[:5000,:-1]), axis=0).astype('float32') # all columns but the last
labels_data = np.concatenate((cat[:5000,-1], mosquito[:5000,-1]), axis=0).astype('float32') # the last column

# train/test split (divide by 255 to obtain normalized values between 0 and 1)
features_train, features_test, labels_train, labels_test = train_test_split(X/255.,y,test_size=0.5,random_state=0)

# one hot encode outputs
labels_train_cnn = np_utils.to_categorical(labels_train)
labels_test_cnn = np_utils.to_categorical(labels_test)
num_classes = labels_test_cnn.shape[1]
# reshape to be [samples][pixels][width][height]
features_train_cnn = features_train.reshape(features_train.shape[0], 1, 28, 28).astype('float32')
features_test_cnn = features_test.reshape(features_test.shape[0], 1, 28, 28).astype('float32')

