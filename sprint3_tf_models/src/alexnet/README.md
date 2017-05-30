# Discussion of AlexNet (2012)

- This collection headed us towards AlexNet in the first place: [Overview by Adit Deshpande](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)

- Link to the paper: [Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

# General evolution: From MNIST to CIFAR10/100 to ImageNet

- Dataset from the [LSVRC-2010 challenge](http://image-net.org/challenges/LSVRC/2010/)

[Training Inception](https://github.com/tensorflow/models/blob/master/inception/README.md#getting-started)

[About Tensorflow and tf slim](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/10/30/image-classification-and-segmentation-using-tensorflow-and-tf-slim/)

[One werid trick paper](https://arxiv.org/abs/1404.5997)

[Relationship between fully connected layers and convolutional layers](https://www.quora.com/How-does-the-conversion-of-last-layers-of-CNN-from-fully-connected-to-fully-convolutional-allow-it-to-process-images-of-different-size)

[About Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

[Striving for simplicity: the all convolutional net](https://arxiv.org/pdf/1412.6806.pdf)

[Comment of Yann LeCun](https://www.facebook.com/yann.lecun/posts/10152820758292143)


Wir wissen nicht, wie das Flowers Dataset überhaupt aussieht. Daher sollte es in Tensorboard exportiert werden.
Embedding: Feature Map Repräsentationen, kleine Abbildungen als Bilder.

Benutze AlexNet (eventuell findet sich da ein vortrainiertes) für diesen Task. 5 KLassen sind echt wenig.

Wo sind die summaries versteckt? -> können durch `end_points` / übergebene Operationen selbst erstellt werden

[README for slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)

The [nets directory from the slim core library (not models)](https://github.com/tensorflow/tensorflow/tree/r0.11/tensorflow/contrib/slim/python/slim/nets) was not included in the package build in **r1.0**. However, it is included singe **r1.1** [see this code line in the BUILD file](https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/contrib/slim/BUILD#L65).

I receive the following error:

```python
In [46]: dataset = flowers.get_split('validation', "/media/niklas/lin-win-hdd/nn_datasets/image/flowers/")
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-46-6def5b1bd741> in <module>()
----> 1 dataset = flowers.get_split('validation', "/media/niklas/lin-win-hdd/nn_datasets/image/flowers/")

/home/niklas/ML-KA/tensorflow/models/slim/datasets/flowers.py in get_split(split_name, dataset_dir, file_pattern, reader)
     87   labels_to_names = None
     88   if dataset_utils.has_labels(dataset_dir):
---> 89     labels_to_names = dataset_utils.read_label_file(dataset_dir)
     90
     91   return slim.dataset.Dataset(

/home/niklas/ML-KA/tensorflow/models/slim/datasets/dataset_utils.py in read_label_file(dataset_dir, filename)
    126   labels_filename = os.path.join(dataset_dir, filename)
    127   with tf.gfile.Open(labels_filename, 'r') as f:
--> 128     lines = f.read().decode()
    129   lines = lines.split('\n')
    130   lines = filter(None, lines)

AttributeError: 'str' object has no attribute 'decode'
> /home/niklas/ML-KA/tensorflow/models/slim/datasets/dataset_utils.py(128)read_label_file()
    126   labels_filename = os.path.join(dataset_dir, filename)
    127   with tf.gfile.Open(labels_filename, 'r') as f:
--> 128     lines = f.read().decode()
    129   lines = lines.split('\n')
    130   lines = filter(None, lines)

ipdb> f.read()
'0:airplane\n1:automobile\n2:bird\n3:cat\n4:deer\n5:dog\n6:frog\n7:horse\n8:ship\n9:truck\n'
ipdb>

```

## What to do

- load the dataset from [the slim alexnet implementation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/alexnet.py)
- image preprocessing (resizing) via the [inception preprocessing module](https://github.com/tensorflow/models/blob/master/slim/preprocessing/inception_preprocessing.py#L278)
- train it
- load alexnet
- export other metrics besides the loss
- export to tensorboard

- TF-slim README: [Link](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)

- What did we use?
    - Slim queues
    - inception preprocessing
    - alexnet implementation (Besonderheit: fully connected = conv2d layers)

Lessons Learned:

- Why using **xrange**? [Link](http://pythoncentral.io/how-to-use-pythons-xrange-and-range/)

> What does that mean? Good question! It means that xrange doesn't actually generate a static list at run-time like range does. It creates the values as you need them with a special technique called yielding. This technique is used with a type of object known as generators.

- How to read in data with TensorFlow queues [TensorFlow Reading Data How To](https://www.tensorflow.org/programmers_guide/reading_data)
    - at best: use the TFrecord format

## Preparation for today

## Prerequisites

- Use **Python 2**!
- The net implementations ship beginning with TF 1.1 (not compiled before that)
- Be sure that you are able to use your GPU!

1. Look in the paper for AlexNet
2. Look into the ImageNet Dataset
3. Look into the definition of alexnet in tf-slim: [src](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/alexnet.py)
    - Mit den `arg_scopes` werden Konfigurationen für alle Layer getroffen (z.B. `activation function`, `weights_regularizer`, `biases_initializer`, `padding`)
4. ImageNet takes too long to train
5. Introduce into the Flowers Data Set
6. Take a look at TFRecords and the DataLoader Class
7. Flowers: Heterogeneous images, varying image size, need preprocessing: [e.g. inception preprocessing](https://github.com/tensorflow/models/blob/master/slim/preprocessing/inception_preprocessing.py#L278)
8. port calculations on the GPU instead of the default CPU


## Lessons Learned

- Update operations that shall be updated when specifying the `create_training_ops` (these updates are not needed for training)
- update_ops: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/learning.py#L394

## ToDo

- Export one image per batch with prediction
- Export one image per batch with feature map
- Create image embedding
