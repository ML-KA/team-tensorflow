"""Implementation of AlexNetv2 using the tf.contrib.slim library.

Prepared for the Team TensorFlow Meetup. 
Goals:
- Load a checkpoint from already trained AlexNet
- continue training from that checkpoint
- Export meaningful summaries to tensorflow
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import tensorflow.contrib.slim as slim

class DataLoader:
    # create a class: data Loader

    ## method: download data and specify destination

    ## method: get training batch

    ## method: get test batch


class AlexNet:
    # create a class AlexNet

    ## place default values here centrally

    def inference(self, input):
        ## method: graph part: inference
        pass

    def loss(self):
        ## method: graph part: loss
        pass

    def training(self):
        ## method: graph part: training
        pass

    def metrics(self):
        ## method: graph part: evaluation / accuracy / other evaluation metrics than the loss (which is optimized)
        pass

    def build_graph(self):
        ## method: build graph (use the methods above)
        pass

    def arg_parse(self):
        ## define the argument parser here and hand it to the executing trainNet file.
        #  Arguments may differ largely between network types.
        # put AlexNet is defined completely by  special parameter setting
        pass

    def create_summaries(self):
        ## method: create summaries (maybe solve this with an decorator!
        pass

    def create_embeddings(self):
        ## method: create embedding
        pass

    def create_checkpoint(self):
        ## method: export checkpoints: where to put this?
        pass


class TrainNet:
    # create class TrainNet: should be agnostic to the neural net. dunno if this is possible.

    ## import the model

    ## essentially a main function with an iterator over a batch

    ## specify batch_size, learning rate, no of epochs, optimizer

    ## questionable: specify loss function, activation function, initializations?


