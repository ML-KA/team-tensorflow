#!/bin/bash
source activate tensorflow
python mnist_with_summaries.py --log_dir="./log"
tensorboard --logdir=./log
