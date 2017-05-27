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

# create a class: data Loader

## method: download data and specify destination

## method: get training batch

## method: get test batch


# create a class AlexNet

## place default values here for

## define the argument parser here and hand it to the executing trainNet file. Arguments may differ largely between network types

## method: graph part: inference

## method: graph part: loss

## method: graph part: training

## method: graph part: evaluation / accuracy / other evaluation metrics

## method: build graph (use the methods above)

## method: create summaries (maybe solve this with an decorator!

## method: create embedding

## method: export checkpoints


# create class TrainNet: should be agnostic to the neural net. dunno if this is possible.

## import the model

## essentially a main function with an iterator over a batch

## specify batch_size, learning rate, no of epochs, optimizer

## questionable: specify loss function, activation function, initializations?


