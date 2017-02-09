import argparse
import csv
import tensorflow as tf
import numpy as np
import hasy_tools as hasy
import input_data

def main(args):
	print("loading data")
	data = input_data.read_data_sets(args.train, args.test, one_hot=True)
	print("data loaded")
	  # Create the model
	num_features = 32*32
	num_classes = 369

	with tf.name_scope('data'):
		x = tf.placeholder(tf.float32, [None, num_features])
		y_ = tf.placeholder(tf.float32, [None, num_classes])
		
	with tf.name_scope('weights'):
		W = tf.Variable(tf.zeros([num_features, num_classes]))
		b = tf.Variable(tf.zeros([num_classes]))

	with tf.name_scope('Wx_plus_b'):
		y = tf.matmul(x, W) + b

	with tf.name_scope('loss'):
		cross_entropy = tf.reduce_mean(
			  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	# Define loss and optimizer

	# The raw formulation of cross-entropy,
	#
	#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
	#                                 reduction_indices=[1]))
	#
	# can be numerically unstable.
	#
	# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
	# outputs of 'y', and then average across the batch.
	
	with tf.name_scope('train'):
		train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		summary = tf.summary.FileWriter('./graph', sess.graph)
		# Train
		for i in range(1000):
			batch_xs, batch_ys = data.train.next_batch(100)
			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

		# Test trained model
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) 
		summ = 0
		for _ in range(100):
			batch_xs, batch_ys = data.test.next_batch(100)
			summ += sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
		print(summ/10000)
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
	print("start")
	main(args)