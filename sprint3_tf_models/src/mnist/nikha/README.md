# Inspecting MNIST

## Sources

-	[Structuring Your TensorFlow Models by Danijar Hafner](http://danijar.com/structuring-your-tensorflow-models/)
-	[Official TensorFlow (non-deep) MNIST Tutorial](https://www.tensorflow.org/get_started/mnist/beginners)

## Directory Structure of the Tutorial

```
mnist (logically structured)
├── __init__.py --- makes this directory a python package
├── input_data.py --- wrapper for providing the mnist data
├── mnist_softmax.py --- a minimal mnist implementation, no use of tensorboard
├── mnist_softmax_xla.py --- the above implementation, together with the XLA compiler
├── mnist_with_summaries.py --- exports many things to tensorboard
├── mnist_deep.py --- a deep CNN version w/o summary to tensorboard
├── mnist.py --- model definition of mnist, is used with fully_connected_feed
├── fully_connected_feed.py --- uses mnist.py
└── BUILD --- used by Google's Bazel build tool
```

Link to [Bazel](https://bazel.build/versions/master/docs/build-ref.html).  
My personal favorite: `mnist.py` together with `fully_connected_feed.py`.
Nice code separation of model definition and execution.
But summary exports are missing.
Extend these models with export functionality.

## Code Structure

1. Define the methods:
    - `placeholder_inputs`: Generate placeholder variables to represent the input tensors.
    - `fill_feed_dict`: Fills the feed_dict for training the given step.
    - `do_eval`: Runs one evaluation against the full epoch of data
2. `main`: Parse arguments: max no of steps (or no of epochs), input data dir, batch size, learning rate...
3. `run_training`: First, builds the computational graph by invoking
    - `mnist.inference`: forward pass through the network (hidden1, hidden2, softmax)
    - `mnist.loss`: adds the operation for calculating the loss (cross entropy in this case)
    - `mnist.training`: add the optimizer (e.g. GradientDescent), track the global step (set this variable as not trainable) and create a summary export (see the [summary generation guide](https://www.tensorflow.org/api_guides/python/summary#Generation_of_Summaries))
    - `mnist.evaluations`: Check, whether the targets are correctly predicted with the [in_top_k (with k=1)](https://www.tensorflow.org/api_docs/python/tf/nn/in_top_k) function
    - Do some instantiations (not yet initialized)
    ```python
    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    ```
    - **Initialize** the variables with `sess.run(init)`
    - Do the training with the steps
        - `fill_feed_dict`: get the next batch / inputs for the model
        - `_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)`: runs the operations `train_op` and `loss`
        - write the summaries and save the **checkpoints** (checkpoints can also be loaded later on)
        - print model evaluations periodically

### Comments on the example code

- **name_scopes** are defined in the model module (`mnist.py`)
- summaries are also defined in the model
- The computational graph is assembled in the **run_training** method (does not seem optimal), maybe one should package it in another method
- Not all values are exported
- Usage of **decorators** for namescopes and summaries would be nice (e.g. see [Danijar Hafners Blog Post    ](http://danijar.com/structuring-your-tensorflow-models/))
- About the **computational graph**: In order to be properly trained, it is enough to do a `sess.run()` with the **sinks of the computational graph**. However, intermediate results can not be saved during execution and printed out for informational purposes. The export of intermediate results via **tf.summary()** is not effected, because it is also an operation in the computational graph. In consequence, **it might sometimes be necessary to apply a sess.run() on some operations individually**. An example:
    -  `sess.run([train_op], feed_dict=feed_dict)` works perfectly fine to train the model. The statement of `train_op` after following the code path is `train_op = optimizer.minimize(loss, global_step=global_step)`, which has the return value ([taken from here](https://www.tensorflow.org/versions/r0.11/api_docs/python/train/optimizers#Optimizer.minimize))
    > An Operation that updates the variables in var_list. If global_step was not None, that operation also increments global_step  

    - if one wants to monitor the loss during training, the evaluation of the `loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')` statement needs to be done individually: `_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)`. This saves the value from `loss` and discards the value from `train_op`.

### Suggested Improvements

- Export more summaries ([summary guide](https://www.tensorflow.org/get_started/summaries_and_tensorboard))
- Do not delete the `logdir` with the checkpoints at start, but place a checkpoint alongside the existent one
- Use meaningful checkpoint names, or do somehow set metadata for the checkpoints. E.g., compare models with different learning rates
- Develop the `mnist.py` module to a general `multilayer-perceptron` module
    - Pass a list of hidden layers
    - pass a list of activation functions
- The precision is not reported, only the (cross-entropy) **loss**. The loss is not in [0,1]. Evaluation is done only with the `do_eval` method

### Lessons Learned

Structure the methods like the following:
1. Define the model in a separate module
    - use the methods `inference, loss, training, evaluations` with possible submethods
    - Use Decorators for namespaces (every function should be a namespace) [**this might not be possible all the time, e.g. when the layers are not easily separable like in the MLP setting**]
    - Use decorators to send variables to summary
    - Use a dictionary to parametrize: learning rate, activation functions, no of layers, no of hidden units, ...
2. Write a custom **data loader** for the dataset (might be necessary most of the time) to generate the batches
3. Write an **execution module** that could be used for nearly any model, as long as the model class and the feeder (data loader) is given. It should consist of the following methods:
    - `main`: to read in the provided arguments
    - `assemble graph`: assemble the graph with the methods from the model module
    - `training`: a foor loop with training steps
    - `create_checkpoints`: to create checkpoints and summaries periodically
4. Define one directory for every model in the common `logdir`. Tensorboard can handle the distributed checkpoints, as long as the parent directory is specified as `--logdir`
5. **The Main Function:** The [argparse module](https://docs.python.org/2/library/argparse.html) is used to pass arguments to the command line.[`FLAGS, unparsed = parse_known_args`](https://docs.python.org/3.4/library/argparse.html#partial-parsing) can be used to parse the provided arguments only partially. This can be useful if the unneeded arguments in `unparsed` are passed to another program inside the current one.
6. As stated in the [documentation](https://www.tensorflow.org/versions/r0.10/get_started/basic_usage#interactive_usage), the `InteractiveSession` class is provided for ease of use in interactive environments such as `IPython`. Therefore, I consider it not to be good coding style to use it in a module. Instead, one should use [`sess.run`](https://www.tensorflow.org/api_docs/python/tf/Session#run) to execute the functions.

## Visualization and Interpretation

One should make extensive use of summary exports and the [embedding](https://www.tensorflow.org/get_started/embedding_viz). But keep disk space consumption in mind.

[Visualizing MNIST Source](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)

Or, [following the original tutorial](https://www.tensorflow.org/get_started/embedding_viz):
>If you have images associated with your embeddings, you will need to produce a single image consisting of small thumbnails of each data point. This is known as the sprite image. The sprite should have the same number of rows and columns with thumbnails stored in row-first order: the first data point placed in the top left and the last data point in the bottom right:

## Next Week

Proposals:
- ~~[word2vec](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/word2vec)~~
- ~~[deepdream](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream) (is a Jupyter Notebook)~~
- ~~[Examples by Aymeric Damien](https://github.com/aymericdamien/TensorFlow-Examples/tree/master/examples)~~
- Deep MNIST: [Tutorial](https://www.tensorflow.org/get_started/mnist/pros), [Source Code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py)
- AlexNet: [AlexNet_v2 tensorflow/models slim implementation](https://github.com/tensorflow/models/blob/master/slim/nets/alexnet.py), [Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- [A great and short guide to CNNs](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)
- [Proposal: Paper list for upcoming Models](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)
