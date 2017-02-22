from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import random
import sys
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from collections import Counter


from tensorflow.contrib.tensorboard.plugins import projector

# choose most common 10.000 words.
# build co-occurrence matrix window size of 3 centered at current word
# calc SVD on co-occurrence matrix using tf.svd
# calc word embeddings
VOCAB_SIZE = 1000
embed_size = 128
graph_dir = 'graph_embedding_pca/'
def main(args):
    data = __load_data(args.data) #2D matrix
    print(data.shape)
    
    w2id = build_vocab(data, VOCAB_SIZE)
    
    cooc = np.zeros([VOCAB_SIZE, VOCAB_SIZE], dtype=np.float32)

    UNK = 0
    for i in xrange(len(data) - 1):
        n = w2id.get(data[i + 1], UNK)
        c = w2id.get(data[i], UNK)
        cooc[n][c] = cooc[n][c] + 1
        cooc[c][n] = cooc[c][n] + 1

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        embedding_matrix = getembedding(sess, cooc, embed_size)
        final_embedding_matrix = sess.run(embedding_matrix)
        embedding_var = tf.Variable(final_embedding_matrix[:1000], name='embedding')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter(graph_dir)

        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        embedding.metadata_path = 'data/vocab_lsa_1000.tsv'

        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, graph_dir + 'model.ckpt', 1)



def build_vocab(data, vocab_size):
    w2id = dict()
    count = [('UNK', -1)]
    count.extend(Counter(data).most_common(vocab_size - 1))
    print("%d different words" % len(count))

    idx = 0;
    with open('data/vocab_lsa_1000.tsv', "w") as f:
        for word, _ in count:
            w2id[word] = idx
            if idx < 1000:
                f.write(word + '\n')
            idx += 1
    return w2id

# k is embed_size
def getembedding(sess, cooc, embed_size):
    s = tf.svd(cooc)
    sess.run(s)
    embed_size = min(embed_size, s[1].get_shape()[1].value, s[2].get_shape()[0].value)
    embedding  = tf.matmul(s[2][:embed_size], tf.transpose(cooc)) # embedding vectors in columns
    return tf.transpose(embedding)




	
def __load_data(file):
	data = []
	with open(file, 'r') as f:
		reader = csv.reader(f, delimiter=" ")
		for line in reader:
			data = np.array(line)
	return data




def __get_parser():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("data", type=str)
	return parser

if __name__ == "__main__":
	args = __get_parser().parse_args()
	main(args)