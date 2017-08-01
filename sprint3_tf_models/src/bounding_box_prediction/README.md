# Guide For Starting Bounding Box Prediction With Tensorflow

We rely on [the TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection)

## Train a Model Locally

- Follow the [Instructions to run a model locally](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_locally.md)
- To do this, one needs to, [install the object detection API](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md#protobuf-compilation)
- The implementation ships with 2 datasets, PASCAL VOC and Oxford-IIIT-Pets. Download the dataset of interest and [prepare the datasets in a TFRecord Format](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/preparing_inputs.md). The pet dataset is smaller.
- One needs to configure the [Object Detection Pipeline](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/configuring_jobs.md). Luckily, there are some sample configuration files available: [Sample Pipeline Config Files](https://github.com/tensorflow/models/tree/master/object_detection/samples/configs).
- Fine-tuning from a checkpoint is recommended. But which model to use? [Choose a checkpoint from the model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md): `rfcn_resnet101_coco` seems like a reasonable trade-off between score and speed. The `faster_rcnn_inception_resnet` needs more than 5GB of VRAM.
- Then [run the training job](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_locally.md#running-the-training-job)
- After some iterations (and saved checkpoints!), make an [evaluation of the fine-tuned model](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_locally.md#running-the-evaluation-job)
- You might want to [apply the model in this notebook](). It might be not that straightforward as you might need to generate a *frozen inference graph*.

## Follow the Source

What happens when you train a model?

- [train script](https://github.com/tensorflow/models/blob/master/object_detection/train.py#L192-L194)
- Builds a model via [model_builder](https://github.com/tensorflow/models/blob/master/object_detection/train.py#L147-L150)
- [trainer: train method, invokes slim.training](https://github.com/tensorflow/models/blob/master/object_detection/trainer.py#L138-L296)
- [Faster R-CNN meta architecture](https://github.com/tensorflow/models/blob/master/object_detection/meta_architectures/faster_rcnn_meta_arch.py) might give insights in how the model works: Drawing bounding boxes, feature extraction.
