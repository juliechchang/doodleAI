### Preprocessing for CNN Implementation

Takes .npz file, and processes into array of corresponding .npy files. The second and third files must be run in Python3, v3.4+. We preprocess the data by reading in .npz files, which contain vector-based stroke-drawing data. We use draw_strokes to save these as svg images, which then then process in to png images, which we then convert to grayscale bitmap and save in .npy format. We save the current drawings in 48 x 48 size, with colors inverted for faster computation. In converting the png to grayscale bitmap, we scale the images (which vary in size after vector-based reproduction) centered on an empty background.

The helper file contains functions for displaying the various types of images. 

The scripts will save and erase files as it processes.
