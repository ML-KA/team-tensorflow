#!/bin/bash
python mnist_with_summaries.py --log_dir="./log"
tensorboard --logdir=./log
