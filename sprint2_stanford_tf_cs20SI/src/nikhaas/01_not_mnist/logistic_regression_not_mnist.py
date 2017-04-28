""""
Edited MNIST logistic regression to predict the notMNIST
data with TensorFlow

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import json
import os

# Define params for the model

with open('config.json') as config_file:
    config = json.load(config_file)

learning_rate = config['learning_rate']
batch_size = config['batch_size']
n_epochs = config['n_epochs']

# Step 1: Read in data
not_mnist = input_data.read_data_sets('notMNIST_data', one_hot=True)

# Step 2: create placeholders for features and labels
# each image in the notMNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, correspondong to letters A-J
# labels follow a one-hot-encoding
X = tf.placeholder(tf.float32, [batch_size, 784], name='X_placeholder')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='Y_placeholder')

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([1, 10]), name='bias')

# Step 4: Build model
# the model that returns the logits
# this logit will be later passed though a softmax layer
logits = tf.matmul(X, w) + b

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y,
                                                  logits=logits, name='loss')

loss = tf.reduce_mean(entropy)

# Step 6: define training op.
optimizer_list = {
    "gradient_descent": lambda learn_rate, loss:
    tf.train.GradientDescentOptimizer(learn_rate).minimize(loss),
    "adadelta": lambda learn_rate, loss:
    tf.train.AdadeltaOptimizer(learn_rate).minimize(loss),
    "adam": lambda learn_rate, loss:
    tf.train.AdamOptimizer(learn_rate).minimize(loss)
}

for key, value in config['optimizer'].items():
    if value:
        optimizer_func = optimizer_list[key]

optimizer = optimizer_func(learning_rate, loss)

with tf.Session() as sess:

    output = {"train_samples": -1,
              "test_samples": -1,
              "accuracy": -1,
              "time": -1
              }
    # to visualize using TensorBoard
    writer = tf.summary.FileWriter('./graph', sess.graph)

    # print('samples: train {0} \n test {1}'
    #       .format(not_mnist.train.num_examples,
    #               not_mnist.test.num_examples))
    output["train_samples"] = not_mnist.train.num_examples
    output["test_samples"] = not_mnist.test.num_examples

    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(not_mnist.train.num_examples / batch_size)

    for i in range(n_epochs):
        total_loss = 0

        for _ in range(n_batches):
            X_batch, Y_batch = not_mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss],
                                     feed_dict={X: X_batch, Y: Y_batch})
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))

    print('Optimization Finished!')

    # Test the model
    n_batches = int(not_mnist.test.num_examples / batch_size)

    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, Y_batch = not_mnist.test.next_batch(batch_size)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits],
                                               feed_dict={X: X_batch,
                                                          Y: Y_batch})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)

    print('Accuracy {0}'.format(total_correct_preds /
                                not_mnist.test.num_examples))
    output["accuracy"] = total_correct_preds / not_mnist.test.num_examples
    output["time"] = time.time() - start_time

    # Write results in a file
    agg_outputs = []

    if not os.path.exists('results.json'):
        with open('results.json', 'w+') as f:
            f.write('[]')

    with open('results.json', 'r') as output_file:
        agg_outputs = json.load(output_file)

    # new Python 3.5 syntax
    config_and_result = {**config, **output}
    agg_outputs.append(config_and_result)

    # Pretty Print the results
    with open('results.json', 'w') as output_file:
        json.dump(agg_outputs, output_file, sort_keys=True,
                  indent=4, separators=(',', ': '))

    writer.close()
    sess.close()
