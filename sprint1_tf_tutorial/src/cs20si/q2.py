import argparse
import csv
import tensorflow as tf
import numpy as np

def main(args):
	x_train, y_train, x_test, y_test = _load_data(args.data, hot_one=True)
	x = tf.placeholder(tf.float32, [None, 9])
	W = tf.Variable(tf.zeros([9, 2]))
	b = tf.Variable(tf.zeros([2]))
	y = tf.matmul(x, W) + b

	y_ = tf.placeholder(tf.float32, [None, 2])
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	sess.run(train_step, feed_dict={x: x_train, y_: y_train})
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
	acc = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
	print(acc)

def _load_data(filepath, hot_one):
	with open(filepath, "r") as fp:
		reader = csv.reader(fp, delimiter='\t')
		idx = -1
		classes = []
		num_samples = 462
		num_classes = 2
		num_features = 9
		string_col = 4
		all_data = []
		for line in reader:
			if (idx == -1):
				classes = line
			else:
				all_data.append(line)
			idx = idx + 1

	np.random.shuffle(all_data)
	#print(all_data)

	data = np.zeros((num_samples, num_features))
	labels = np.zeros(num_samples, dtype=np.int32)
	for i in range(num_samples):
		labels[i] = int(all_data[i][-1])
		data[i,:string_col] = [float(x) for x in all_data[i][:string_col]] # col 5 is text
		data[i,string_col+1 :] = [float(x) for x in all_data[i][string_col + 1:-1]] # col 5 is text
		data[i,string_col] = all_data[i][string_col] == "Absent"
	print('data shape: %s; labels shape: %s' % (data.shape, labels.shape))

	num_test_samples = int(num_samples / 5)
	x_test = data[:num_test_samples, :]
	y_test = labels[:num_test_samples]
	x_train = data[num_test_samples:, :]
	y_train = labels[num_test_samples:]

	print('train data shape: %s; labels shape: %s' % (x_train.shape, y_train.shape))
	print('test data shape: %s; labels shape: %s' % (x_test.shape, y_test.shape))

	if (hot_one):
		y_train = _get_hot_one(y_train, 2)
		y_test = _get_hot_one(y_test, 2)
	return x_train, y_train, x_test, y_test

def _get_hot_one(label_array, num_classes):
	num_labels = label_array.shape[0]
	tmp = np.zeros([num_labels, num_classes])
	index_offset = np.arange(num_labels) * num_classes
	tmp.flat[index_offset + label_array] = 1
	return tmp

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, help='data file path')
	args = parser.parse_args()
	main(args)