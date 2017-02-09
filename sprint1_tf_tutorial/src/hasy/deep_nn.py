import argparse
import csv
import tensorflow as tf
import numpy as np
import hasy_tools as hasy
import input_data

def main(args):
	#data = hasy._load_csv(args.data)
	'''
	print("generating index...")
	index = hasy.generate_index('symbols.csv')
	print("total labels %d" % len(index))
	num_classes = len(index)

	
	print("loading training data...")
	x_train, y_train = hasy.load_images(args.train, index, one_hot=False, flatten=True) # image = (-1, pixels) labels=(-1, hot one)
	print("hot one...")
	y_train = _get_hot_one(y_train, num_classes)
	
	print("loading testing data...")
	x_test, y_test = hasy.load_images(args.test, index, one_hot=False, flatten=True) # image = (-1, pixels) labels=(-1, hot one)
	print("hot one...")
	y_test = _get_hot_one(y_test, num_classes)
	'''
	print("loading data")
	data = input_data.read_data_sets(args.train, args.test)
	print("data loaded")
	  # Create the model
	num_features = 32*32
	num_classes = 369

	with tf.name_scope('data'):
		x = tf.placeholder(tf.float32, [None, num_features])
		y_ = tf.placeholder(tf.float32, [None, num_classes])
		

	with tf.name_scope('conv1'):
		W_conv1 = weight_variable([5, 5, 1, 32])
		b_conv1 = bias_variable([32])
		x_image = tf.reshape(x, [-1,32,32,1])

		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)

	with tf.name_scope('conv2'):
		W_conv2 = weight_variable([5, 5, 32, 64])
		b_conv2 = bias_variable([64])

		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)


	with tf.name_scope('fully connected layer'):
		W_fc1 = weight_variable([8 * 8 * 64, 1024])
		b_fc1 = bias_variable([1024])

		h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


	with tf.name_scope('output layer'):
		W_fc2 = weight_variable([1024, num_classes])
		b_fc2 = bias_variable([num_classes])

		y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

	with tf.name_scope('loss'):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	accuracy2 = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))


	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	summary = tf.summary.FileWriter('./graph', sess.graph)

	sess.run(tf.global_variables_initializer())
	for i in range(20000):
		batch = data.train.next_batch(50)
		if i%100 == 0:
		  train_accuracy = accuracy.eval(feed_dict={ x:batch[0], y_: batch[1]})
		  print("step %d, training accuracy %g"%(i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], y_: batch[1]})

	test_accuracy = 0
	for i in range(200):
		batch = mnist.test.next_batch(100)
		test_accuracy += accuracy.eval(feed_dict={x : batch[0], y_:batch[1]})

	print("test accuracy %g"% (test_accuracy/20000))
	

	summary.close()

	print("program finished")


def _get_hot_one(label_array, num_classes):
	num_labels = label_array.shape[0]
	tmp = np.zeros([num_labels, num_classes])
	index_offset = np.arange(num_labels) * num_classes
	tmp.flat[index_offset + label_array] = 1
	return tmp

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('train', type=str, help='data file path')
	parser.add_argument('test', type=str, help='data file path')
	args = parser.parse_args()
	main(args)