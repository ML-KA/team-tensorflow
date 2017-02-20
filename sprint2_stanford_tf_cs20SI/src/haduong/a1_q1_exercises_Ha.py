"""
Simple TensorFlow exercises
You should thoroughly test your code
"""

import tensorflow as tf

###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
outTensor = tf.cond(tf.greater(x, y), lambda: tf.add(x, y), lambda: tf.sub(x, y))

with tf.Session() as sess:
    print(sess.run(outTensor))

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from -1 and 1.
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

x = tf.random_uniform([], minval=-1, maxval=1, name = 'x')
y = tf.random_uniform([], minval=-1, maxval=1, name = 'y')
f1 = tf.add(x,y, name = 'add')
f2 = tf.sub(x,y, name = 'sub')
f3 = tf.zeros([], name = '0')
outTensor = tf.case({tf.less(x,y) : lambda: f1, tf.less(y,x) : lambda: f2}, default = lambda: f3, name = 'out')

with tf.Session() as sess:
    result = sess.run(outTensor)
    print(x)
    print(y)
    print(sess)
    print(outTensor)
    # right call
    print(result)

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]] 
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

x = tf.constant([[0, -2, -1], [0, 1, 2]], name='x')
y = tf.zeros_like(x, name='y')
#outTensor = tf.cond(tf.equal(x,y), lambda: f1, lambda: f2, name='out')
outTensor = tf.equal(x,y)

with tf.Session() as sess:
    print(sess.run(outTensor))

###############################################################################
# 1d: Create the tensor x of value 
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

x = tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
                 30.97266006,  26.67541885,  38.08450317,  20.74983215,
                 34.94445419,  34.45999146,  29.06485367,  36.01657104,
                 27.88236427,  20.56035233,  30.20379066,  29.51215172,
                 33.71149445,  28.59134293,  36.05556488,  28.66994858])
outTensor1 = tf.where(tf.greater(x,30.0))
outTensor2 = tf.gather(params=x, indices=outTensor1)

with tf.Session() as sess:
    print(sess.run(outTensor1))
    print(sess.run(outTensor2))

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

x = tf.diag(diagonal=tf.range(1,7))

with tf.Session() as sess:
    print(sess.run(x))

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

x = tf.diag(tf.random_uniform([10]))

outTensor = tf.matrix_determinant(x)

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(outTensor))

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

x = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
(y, idx)  = tf.unique(x)

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(y))

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.less() and tf.select() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

x = tf.random_uniform([300])
y = tf.random_uniform([300])

def huber_loss(labels, predictions, delta=1.0):
 residual = tf.abs(predictions - labels)
 condition = tf.less(residual, delta)
 small_res = 0.5 * tf.square(residual)
 large_res = delta * residual - 0.5 * tf.square(delta)
 return tf.select(condition, small_res, large_res)

outTensor = huber_loss(x,y)

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(y))
    print(sess.run(outTensor))

