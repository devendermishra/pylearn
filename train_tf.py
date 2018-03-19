import os
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf

def imread(path, mode):
	img = scipy.misc.imread(path, mode=mode).astype(np.float)
	return img


import sys

cwd = sys.argv[1]
path = cwd + "/256_ObjectCategories"
valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

categories = sorted(os.listdir(path))
ncategories = len(categories)
imgs = []
labels = []

# LOAD ALL IMAGES
count = 0
selected_cat = 257
#selected_cat = 25
selected_image = 10
# For initial, let us select 10 only.
import scipy.ndimage.interpolation
import skimage

for i, category in enumerate(categories):
	ctr = 0
	for f in os.listdir(path + "/" + category):
		fullpath = os.path.join(path + "/" + category, f)
		img = scipy.misc.imresize(imread(fullpath, 'RGB'), [32, 32, 3])
		img = skimage.img_as_float(img)
		imgs.append(img)
		labels.append(i)

		ctr = ctr + 1
		if ctr >= selected_image:
			break

	count = count + 1
	if count >= selected_cat:
		break


imgs = np.array(imgs, dtype="float32")
print("Image shape " + `imgs.shape` + " and type " + `imgs.dtype`)
print("Num imgs: %d" % (len(imgs)))
print("Num labels: %d" % (len(labels)))
print(ncategories)
print("First image: " + `imgs[0]`)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.5)

print("Num train_imgs: %d" % (len(X_train)))
print("Num test_imgs: %d" % (len(X_test)))

print("train shape: " + `X_train.shape`)
print("test shape: " + `X_test.shape`)

sess = tf.Session()
batch_size = 50
# batch_size = 15
# evaluation_size = 15
output_every = 10
generations = 2000
generations = 100
eval_every = 2
image_height = X_train[0].shape[0]
image_width = X_train[0].shape[0]
crop_height = 24
crop_width = 24
num_channels = 1
num_targets = selected_cat

min_after_dequeue = 50
capacity = 3 * batch_size + min_after_dequeue

learning_rate = 0.01
lr_decay = 0.9
num_gens_to_wait = 5.

image_vec_length = image_height * image_width * num_channels
record_length = 1 + image_vec_length

x_input_shape = (batch_size, image_width, image_height, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.uint32, shape=(batch_size,))

import random


def _random_selection(test, target, batch_size):
	examples, labels = zip(*random.sample(list(zip(test, target)), batch_size))
	return (np.array(examples), labels)


def input_pipeline(batch_size, test):
	if test:
		return _random_selection(X_test, y_test, batch_size)
	return _random_selection(X_train, y_train, batch_size)


def cnn_model(input_images, batch_size):
	def truncated_normal_var(name, shape, dtype):
		return (tf.get_variable(name=name, shape=shape, dtype=dtype,
		                        initializer=tf.truncated_normal_initializer(stddev=0.05)))

	def zero_var(name, shape, dtype):
		return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

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
	pool1 = tf.nn.max_pool(relu_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer1')
	# Local Response Normalization
	norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm1')

	# Second Convolutional Layer
	with tf.variable_scope('conv2') as scope:
		# Conv kernel is 5x5, across all prior 64 features and we create 64 more features
		conv2_kernel = truncated_normal_var(name='conv_kernel2', shape=[5, 5, 64, 64], dtype=tf.float32)
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
		full_bias1 = zero_var(name='full_bias1', shape=[384], dtype=tf.float32)
		full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, full_weight1), full_bias1))

	# Second Fully Connected Layer
	with tf.variable_scope('full2') as scope:
		# Second fully connected layer has 192 outputs.
		full_weight2 = truncated_normal_var(name='full_mult2', shape=[384, 192], dtype=tf.float32)
		full_bias2 = zero_var(name='full_bias2', shape=[192], dtype=tf.float32)
		full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))
	# Final Fully Connected Layer -> 10 categories for output (num_targets)

	with tf.variable_scope('full3') as scope:
		# Final fully connected layer has 10 (num_targets) outputs.
		full_weight3 = truncated_normal_var(name='full_mult3', shape=[192, num_targets], dtype=tf.float32)
		full_bias3 = zero_var(name='full_bias3', shape=[num_targets], dtype=tf.float32)
		final_output = tf.add(tf.matmul(full_layer2, full_weight3), full_bias3)

	return (final_output)


def training_loss(logits, targets):
	# Get rid of extra dimensions and cast targets into integers
	targets = tf.squeeze(tf.cast(targets, tf.int32))
	# Calculate cross entropy from logits and targets
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
	# Take the average loss across batch size
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	return (cross_entropy_mean)


def train_step(loss_value, generation_num):
	# Our learning rate is an exponential decay (stepped down)
	model_learning_rate = tf.train.exponential_decay(learning_rate, generation_num, num_gens_to_wait, lr_decay,
	                                                 staircase=True)
	# Create optimizer
	my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
	# Initialize train step
	train_step = my_optimizer.minimize(loss_value)
	return (train_step)


def accuracy_of_batch(logits, targets):
	# Make sure targets are integers and drop extra dimensions
	targets = tf.squeeze(tf.cast(targets, tf.int32))
	# Get predicted values by finding which logit is the greatest
	batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
	# Check if they are equal across the batch
	predicted_correctly = tf.equal(batch_predictions, targets)
	# Average the 1's and 0's (True's and False's) across the batch size
	accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
	return (accuracy)


images, targets = input_pipeline(batch_size, test=False)
test_images, test_targets = input_pipeline(batch_size, test=True)

with tf.variable_scope('model_definition') as scope:
	# Declare the training network model
	model_output = cnn_model(images, batch_size)
	# Use same variables within scope
	scope.reuse_variables()
	# Declare test model output
	test_output = cnn_model(test_images, batch_size)

loss = training_loss(model_output, targets)
accuracy = accuracy_of_batch(test_output, test_targets)
generation_num = tf.Variable(0, trainable=False)
train_op = train_step(loss, generation_num)

init = tf.global_variables_initializer()
#saver = tf.train.Saver()
sess.run(init)
#save_path = saver.save(sess, "./tf.train2.save")
#tf.train.start_queue_runners(sess=sess)

train_loss = []
test_accuracy = []
for i in range(generations):
	_, loss_value = sess.run([train_op, loss])
	if (i + 1) % output_every == 0:
		train_loss.append(loss_value)
	#save_path = saver.save(sess, "./tf.train2.save")
	output = 'Generation {}: Loss = {:.5f}'.format((i + 1),
	                                               loss_value)
	print(output)
	if (i + 1) % eval_every == 0:
		[temp_accuracy] = sess.run([accuracy])
		test_accuracy.append(temp_accuracy)
		acc_output = ' --- Test Accuracy={:.2f}%.'.format(100. * temp_accuracy)
		print(acc_output)

eval_indices = range(0, generations, eval_every)
output_indices = range(0, generations, output_every)
# Plot loss over time
plt.plot(output_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()
plt.savefig("loss.png")
# Plot accuracy over time
plt.plot(eval_indices, test_accuracy, 'k-')
plt.title('Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.show()
plt.savefig("accuracy.png")

#tf.initialize_all_variables()
#sess.run([train_op, loss, tf.initialize_all_variables(), tf.global_variables_initializer()])
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
save_path = saver.save(sess, "./tf.train2.save")
# print("Saved at " + save_path)
