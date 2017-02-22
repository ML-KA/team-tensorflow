from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np 
import sys
import argparse
from process_data import process_data
import math
import os
import time

from tensorflow.contrib.tensorboard.plugins import projector

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000 # how many steps to skip before reporting the loss

WEIGHTS_FLD = 'processed/'

class SkipGramModel:

    def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.center_words = tf.placeholder(tf.int32, shape=[None], name='center_words')
            self.target_words = tf.placeholder(tf.int32, shape=[None, 1], name='target_words')

    def _create_embedding(self):
        with tf.name_scope("embed"):
            self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0))

    def _create_loss(self):
        # define inference
        with tf.name_scope("loss"):
            embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')
            # construct variables for NCE(noise contrastive estimation) loss
            nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size], stddev=1.0/math.sqrt(self.embed_size)))
            nce_bias = tf.Variable(tf.zeros([self.vocab_size], name='nce_bias'))
            #bias = tf.reshape(nce_bias, [1, -1])
            #output = tf.add(tf.matmul(embed,tf.transpose(nce_weight)), bias) # 128 x 50000
            #_, indices = tf.nn.top_k(output, 5)
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, labels=self.target_words, inputs=embed, num_sampled=self.num_sampled, num_classes=self.vocab_size), name='loss')
        #return self.loss, indices

    def _create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()


    def _build_graph(self):
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

  
def train_model(model, batch_gen, num_train_steps, weights_fld):

    saver = tf.train.Saver()
    initial_step = 0
    #config = tf.ConfigProto()
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    
    #with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # restore latest checkpoint: mapping of variable names to tensors
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path :
            saver.restore(sess, ckpt.model_checkpoint_path)


        total_loss = 0.0
        writer = tf.summary.FileWriter('improved_graph/lr' + str(model.learning_rate), sess.graph)
        initial_step = model.global_step.eval()
        for index in xrange(initial_step, initial_step + num_train_steps):
            centers, targets = batch_gen.next()
            loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict={model.center_words: centers, model.target_words:targets})
            writer.add_summary(summary, global_step=index)
            writer.flush()
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / SKIP_STEP))
                total_loss = 0.0
                saver.save(sess, 'checkpoints/skip-gram', global_step=model.global_step)
        writer.close()
        '''
        while True:
            input = raw_input('Enter word \n')
            if input == 'exit':
                break
            in_id = dict.get(input, 0)
            y = sess.run(indices, feed_dict={model.center_words: [in_id]})
            for i in xrange(y.shape[1]):
                print('%s' %index_dict[y[0][i]])
        '''
        ####################
        # code to visualize the embeddings. uncomment the below to visualize embeddings
        final_embed_matrix = sess.run(model.embed_matrix)
        
        # # it has to variable. constants don't work here. you can't reuse model.embed_matrix
        embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter('processed')

        # # add embedding to the config file
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        
        # # link this tensor to its metadata file, in this case the first 500 words of vocab
        embedding.metadata_path = 'data/vocab_1000.tsv'

        # # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, 'processed/model3.ckpt', 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--vocab_size', type=int, default=VOCAB_SIZE)
    parser.add_argument('--embed_size', type=int, default=EMBED_SIZE)
    parser.add_argument('--num_sampled', type=int, default=NUM_SAMPLED)
    parser.add_argument('--skip_window', type=int, default=SKIP_WINDOW)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--data', type=str)
    args = parser.parse_args()

    print("create model")
    model = SkipGramModel(args.vocab_size, args.embed_size, args.batch_size, args.num_sampled, args.learning_rate)
    model._build_graph()
    print("generating batch")
    batch_gen, dict, index_dict = process_data(args.vocab_size, args.batch_size, args.skip_window, args.data)
    start = time.time()
    start1 = time.clock()
    train_model(model, batch_gen, NUM_TRAIN_STEPS, WEIGHTS_FLD)
    duration = time.time() - start
    duration1 = time.clock() - start1
    print("elapsed time for training: %f, %f" % (duration, duration1))