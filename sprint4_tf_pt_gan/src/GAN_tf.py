import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import tensorflow as tf

'''
global parameters
'''
g_output_size = 784
g_input_size = 100
g_hidden_size = 128

minibatch_size = 128
num_epochs = 1000000
learning_rate = 1e-3
xavier_init = tf.contrib.layers.xavier_initializer()
zero_init = tf.zeros_initializer()

def discriminator_tf(X, reuse=False):
    with tf.variable_scope('discriminator'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        W1 = tf.get_variable('Dis_W1', [g_output_size, g_hidden_size ],
                             initializer=xavier_init)
                             #tf.random_normal_initializer(stddev=xavier_init([J, K])))
        B1 = tf.get_variable('Dis_B1', [g_hidden_size ], initializer=zero_init)
        W2 = tf.get_variable('Dis_W2', [g_hidden_size , 1],
                             initializer=xavier_init)
                             #tf.random_normal_initializer(stddev=xavier_init([K, L])))
        B2 = tf.get_variable('Dis_B2', [1], initializer=zero_init)

        # summary
        tf.summary.histogram('weight1', W1)
        tf.summary.histogram('weight2', W2)
        tf.summary.histogram('biases1', B1)
        tf.summary.histogram('biases2', B2)

        Dis_h1 = tf.nn.elu((tf.matmul(X, W1) + B1))
        Dis_h1_dropout = tf.nn.dropout(Dis_h1, 0.5)
        Dis_logits = tf.matmul(Dis_h1_dropout, W2) + B2

        return Dis_logits

def sample_Z_tf(minibatch_size, g_input_size):
    return np.random.uniform(-1.0, 1.0, size=[minibatch_size, g_input_size]).astype(np.float32)

def generator_tf(z):
    with tf.variable_scope('generator'):


        W1 = tf.get_variable('Gen_W1', [100, g_hidden_size ],
                             initializer=xavier_init)

        B1 = tf.get_variable('Gen_B1', [g_hidden_size ], initializer=zero_init)
        W2 = tf.get_variable('Gen_W2', [g_hidden_size , g_output_size],
                             initializer=xavier_init)

        B2 = tf.get_variable('Gen_B2', [g_output_size], initializer=zero_init)

        # summary
        tf.summary.histogram('weight1', W1)
        tf.summary.histogram('weight2', W2)
        tf.summary.histogram('biases1', B1)
        tf.summary.histogram('biases2', B2)

        Gen_h1 = tf.nn.elu((tf.matmul(z, W1) + B1))
        Gen_o = tf.matmul(Gen_h1, W2) + B2
        prob = tf.nn.sigmoid(Gen_o)

        return prob


def load_data():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    return mnist


def plot(samples):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig


def trainGAN():


    mnist = load_data()

    with tf.variable_scope('Placeholder'):
        # Raw image
        X_tf = tf.placeholder(tf.float32, [None, g_output_size])
        tf.summary.image('Raw Image', tf.reshape(X_tf, [-1, 28, 28, 1]), 3)
        # Noise
        Z_tf = tf.placeholder(tf.float32, [None, g_input_size])  # noise
        tf.summary.histogram('Noise', Z_tf)

    with tf.variable_scope('GAN'):
        Gen_tf = generator_tf(Z_tf)

        Dis_real_logits_tf = discriminator_tf(X_tf, reuse=False)
        Dis_fake_logits_tf = discriminator_tf(Gen_tf, reuse=True)
    tf.summary.image('Generated Image', tf.reshape(Gen_tf, [-1, 28, 28, 1]), 3)



    with tf.variable_scope('D_loss'):
        Dis_loss_real_tf = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Dis_real_logits_tf, labels=tf.ones_like(Dis_real_logits_tf)))
        Dis_loss_fake_tf= tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Dis_fake_logits_tf, labels=tf.zeros_like(Dis_fake_logits_tf)))
        Dis_loss_tf = Dis_loss_real_tf + Dis_loss_fake_tf

        tf.summary.scalar('Dis_loss_real', Dis_loss_real_tf)
        tf.summary.scalar('Dis_loss_fake', Dis_loss_fake_tf)
        tf.summary.scalar('Dis_Loss', Dis_loss_tf)

    with tf.name_scope('G_loss'):
        Gen_loss_tf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits
                                (logits=Dis_fake_logits_tf, labels=tf.ones_like(Dis_fake_logits_tf)))
        tf.summary.scalar('Gen_Loss', Gen_loss_tf)

    train_var = tf.trainable_variables()
    theta_D = [var for var in train_var if 'discriminator' in var.name]
    theta_G = [var for var in train_var if 'generator' in var.name]

    with tf.name_scope('train'):
        Dis_optimizer_tf = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Dis_loss_tf, var_list=theta_D)
        Gen_optimizer_tf = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Gen_loss_tf, var_list=theta_G)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('tmp/mnist/1')
    writer.add_graph(sess.graph)

    num_img = 0
    if not os.path.exists('output_tf/'):
        os.makedirs('output_tf/')

    for epoch in range(num_epochs):
        X_minibatch, _ = mnist.train.next_batch(minibatch_size)
        batch_noise = sample_Z_tf(minibatch_size, g_input_size)
        if epoch % 1000 == 0:
            samples = sess.run(Gen_tf, feed_dict={Z_tf: sample_Z_tf(25, g_input_size)})
            fig = plot(samples)
            plt.savefig('out_tf/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
            num_img += 1
            plt.close(fig)

        _, dis_loss_print = sess.run([Dis_optimizer_tf, Dis_loss_tf],
                                   feed_dict={X_tf: X_minibatch, Z_tf: batch_noise})

        _, gen_loss_print = sess.run([Gen_optimizer_tf, Gen_loss_tf],
                                   feed_dict={Z_tf: batch_noise})

        if epoch % 100 == 0:
            s = sess.run(merged_summary, feed_dict={X_tf: X_minibatch, Z_tf: batch_noise})
            writer.add_summary(s, epoch)
            print('epoch:%d g_loss:%f d_loss:%f' % (epoch, gen_loss_print, dis_loss_print))


if __name__ == '__main__':
    trainGAN()
