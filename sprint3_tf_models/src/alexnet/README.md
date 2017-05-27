# Discussion of AlexNet (2012)

- This collection headed us towards AlexNet in the first place: [Overview by Adit Deshpande](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)

- Link to the paper: [Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

General evolution: From MNIST to CIFAR10/100 to ImageNet..

[Training Inception](https://github.com/tensorflow/models/blob/master/inception/README.md#getting-started)

[About Tensorflow and tf slim](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/10/30/image-classification-and-segmentation-using-tensorflow-and-tf-slim/)

[One werid trick paper](https://arxiv.org/abs/1404.5997)

[Relationship between fully connected layers and convolutional layers](https://www.quora.com/How-does-the-conversion-of-last-layers-of-CNN-from-fully-connected-to-fully-convolutional-allow-it-to-process-images-of-different-size)

[About Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

[Striving for simplicity: the all convolutional net](https://arxiv.org/pdf/1412.6806.pdf)

[Comment of Yann LeCun](https://www.facebook.com/yann.lecun/posts/10152820758292143)


Wir wissen nicht, wie das Flowers DAtaset überhaupt aussieht. Daher sollte es in Tensorboard exportiert werden.
Embedding: Feature Map Repräsentationen, kleine Abbildungen als Bilder.

Benutze AlexNet (eventuell findet sich da ein vortrainiertes) für diesen Task. 5 KLassen sind echt wenig.

Wo sind die summaries versteckt? Im slim_namescope? Schaue in diesen Namescope hinein.

[README for slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)
