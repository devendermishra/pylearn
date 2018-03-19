import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import initializers
from keras import backend as K


#K.set_image_dim_ordering('th')

import os
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf

def imread(path, mode):
	img = scipy.misc.imread(path, mode=mode).astype(np.float)
	#if len(img.shape) == 2:
	#	img = np.transpose(np.array([img, img, img]), (2, 0, 1))
	return img

imgs = []
labels = []

num_targets = 256
#num_targets = 25
batch_size = 1
# LOAD ALL IMAGES
count = 0
selected_cat = 256
selected_cat = 25
selected_image = 40

learning_rate = 0.09
lr_decay = 0.9
num_gens_to_wait = 5.

#For initial, let us select 10 only.
import scipy.ndimage.interpolation
import skimage
import sys
fullpath = sys.argv[1]
img = scipy.misc.imresize(imread(fullpath, 'RGB'), [32, 32, 3])
img = skimage.img_as_float(img)
imgs.append(img)  # NORMALIZE IMAGE

imgs = np.array(imgs, dtype="float32")
print("Image shape " + `imgs.shape` + " and type " + `imgs.dtype`)
print("Num imgs: %d" % (len(imgs)))
print("Num labels: %d" % (len(labels)))

def cnn_model(input_images, batch_size):
	def truncated_normal_var(name, shape, dtype):
		return (tf.get_variable(name=name, shape=shape, dtype=dtype,
		                        initializer=tf.truncated_normal_initializer(stddev=0.05)))
	def zero_var(name, shape, dtype):
		return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))


	#print("Input image shape: " + `input_images.shape`)

	# First Convolutional Layer
	with tf.variable_scope('conv1') as scope:
		# Conv_kernel is 5x5 for all 1 channel and we will create 64 features
		conv1_kernel = truncated_normal_var(name='conv_kernel1', shape=[5, 5, 3, 64], dtype=tf.float32)
		# We convolve across the image with a stride size of 1
		conv1 = tf.nn.conv2d(input_images, conv1_kernel, [1, 1, 1, 1], padding='SAME')
		# Initialize and add the bias term
		conv1_bias = zero_var(name='conv_bias1', shape=[64], dtype=tf.float32)
		conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
		# ReLU element wise
		relu_conv1 = tf.nn.relu(conv1_add_bias)

	# Max Pooling
	pool1 = tf.nn.max_pool(relu_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool_layer1')
	# Local Response Normalization
	norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3,beta=0.75, name='norm1')

	# Second Convolutional Layer
	with tf.variable_scope('conv2') as scope:
		# Conv kernel is 5x5, across all prior 64 features and we create 64 more features
		conv2_kernel = truncated_normal_var(name='conv_kernel2',shape=[5, 5, 64, 64], dtype=tf.float32)
		# Convolve filter across prior output with stride size of 1
		conv2 = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
		# Initialize and add the bias
		conv2_bias = zero_var(name='conv_bias2', shape=[64], dtype=tf.float32)
		conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
		# ReLU element wise
		relu_conv2 = tf.nn.relu(conv2_add_bias)

	print("relu_conv2 shape " + `relu_conv2.get_shape()`)

	# Max Pooling
	pool2 = tf.nn.max_pool(relu_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer2')
	# Local Response Normalization (parameters from paper)
	norm2 = tf.nn.lrn(pool2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')

	# Reshape output into a single matrix for multiplication for the fully connected layers
	reshaped_output = tf.reshape(norm2, [batch_size, -1])
	reshaped_dim = reshaped_output.get_shape()[1].value

	# First Fully Connected Layer
	with tf.variable_scope('full1') as scope:
		# Fully connected layer will have 384 outputs.
		full_weight1 = truncated_normal_var(name='full_mult1', shape=[reshaped_dim, 384], dtype=tf.float32)
		full_bias1 = zero_var(name='full_bias1', shape=[384],dtype=tf.float32)
		full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output,full_weight1), full_bias1))

	# Second Fully Connected Layer
	with tf.variable_scope('full2') as scope:
		# Second fully connected layer has 192 outputs.
		full_weight2 = truncated_normal_var(name='full_mult2',shape=[384, 192], dtype=tf.float32)
		full_bias2 = zero_var(name='full_bias2', shape=[192], dtype=tf.float32)
		full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))
		# Final Fully Connected Layer -> 10 categories for output (num_targets)

	with tf.variable_scope('full3') as scope:
		# Final fully connected layer has 10 (num_targets) outputs.
		full_weight3 = truncated_normal_var(name='full_mult3', shape=[192, num_targets], dtype=tf.float32)
		full_bias3 = zero_var(name='full_bias3', shape=[num_targets], dtype=tf.float32)
		final_output = tf.add(tf.matmul(full_layer2, full_weight3), full_bias3)

	return(final_output)


checkpoint_directory = "."
checkpoint_file=tf.train.latest_checkpoint(checkpoint_directory)
tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("tf.train.save.meta")


def zero_var(name):
	return (tf.get_variable(name=name, shape=[1,1], dtype=tf.float32, initializer=tf.constant_initializer(0.0)))


saver = tf.train.Saver()


'''
model_definition/conv1/conv_bias1 (DT_FLOAT) [64]
model_definition/conv1/conv_kernel1 (DT_FLOAT) [5,5,3,64]
model_definition/conv2/conv_bias2 (DT_FLOAT) [64]
model_definition/conv2/conv_kernel2 (DT_FLOAT) [5,5,64,64]
model_definition/full1/full_bias1 (DT_FLOAT) [384]
model_definition/full1/full_mult1 (DT_FLOAT) [4096,384]
model_definition/full2/full_bias2 (DT_FLOAT) [192]
model_definition/full2/full_mult2 (DT_FLOAT) [384,192]
model_definition/full3/full_bias3 (DT_FLOAT) [257]
model_definition/full3/full_mult3 (DT_FLOAT) [192,257]

'''
def truncated_normal_var(name, shape, dtype):
	return (tf.get_variable(name=name, shape=shape, dtype=dtype,
		                    initializer=tf.truncated_normal_initializer(stddev=0.05)))
def zero_var(name, shape, dtype):
	return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))


with tf.Session() as sess:

	saver.restore(sess, "tf.train2.save")

	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	tf.tables_initializer().run()
	#sess.run(tf.initialize_all_variables())
	#sess.run([trainable_variables, variables])
	#tf.initialize_all_variables()
	#sess.run(tf.initialize_all_variables())
	with tf.variable_scope('model_definition') as scope:
		#tf.initialize_all_variables()
		#scope.reuse_variables()
		#sess.run(scope)
		output = cnn_model(imgs, batch_size)
	print("Output " + `output`)
	print("Output shape " + `output.get_shape()`)
	sess.run(output)
	print(output.eval())
	answer = output.eval()
	#print("Max value " + `tf.argmax(answer).eval()`)