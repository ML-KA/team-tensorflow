# Meeting on VGG Net

Date: 13th June 2017

## ToDos Beforehand

>After AlexNet and DeConvNet (as mentioned with ZFnet), the next subject is **VGG Net**.
We are still following the list at [The-9-Deep-Learning-Papers-You-Need-To-Know-About](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)

>**Please bring your notebook. It might be needed for code execution.**

># Please Prepare
- Execute AlexNet on your own Laptop: [team-tensorflow AlexNet](https://github.com/ML-KA/team-tensorflow/tree/master/sprint3_tf_models/src/alexnet)
- Replace AlexNet in the code with the tensorflow slim implementation of VGG (slim ships with this network)
- Read the summary section about VGG Net at [The-9-Deep-Learning-Papers-You-Need-To-Know-About](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)
- Read the Paper at [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)


># More Resources
- [Visualization of Convolutional Arithmetics](https://github.com/vdumoulin/conv_arithmetic)


># Goals:
- Replace AlexNet with VGG and train it on the Flowers Data Set

>Link to community repo:
https://github.com/ML-KA/team-tensorflow


## ToDos today

- Focusing on Code execution on your own machine
- Run the AlexNet Jupyter Notebook (it is possible with the tensorflow CPU version, although it is not very effective)
- Live Demo: Replace AlexNet with VGGNet in the Code + Training (show CPU and GPU performance) and show effect of the batch size on execution time
- Discussion of the Paper

## ToDos next time

- Someone else than me puts effort in the next meeting (in two weeks)
- We deal with Google LeNet
- Checkout an [Inception Checkpoint](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models) and use it to infer some images, e.g. on a subset of the images they have been trained on, what is the [ILSVRC-2012-CLS image classification data set](www.image-net.org/challenges/LSVRC/2012/)
- Or use it on whatever dataset you like

# Resources

- [AN overview about gradient descent optimizers by S. Ruder](http://sebastianruder.com/optimizing-gradient-descent/)
- [A video about the evolution of gradient descent optimizers by S. Raval](https://youtu.be/nhqo0u1a6fw)
