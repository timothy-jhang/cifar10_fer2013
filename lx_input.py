# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import model_landmark
from struct import *
import numpy as np

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 36 # input 48x48 --> 36x36 crop KSJHANG
		 # input 256x256 --> 227x227 KSJHANG

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 7 # NUM_CLASSES=2 for gender classification, KSJHANG
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 24000
#for fold0,...fold4 experiments
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 30000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 30000


def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 48 # fer2013 48 x 48 x 1 
  result.width = 48
  result.depth = 1
  result.no_lx = 144
  lx_bytes = 144 # each four bytes - float32

  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  no_record_bytes = label_bytes + image_bytes + lx_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=no_record_bytes)
  result.key, value = reader.read(filename_queue)
  print('>> value = ', value)
#
  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)
  print('>>> value = ', value)
#?? Convert from a string to a vector of uint8 that is record_bytes long.
#??  record_bytes = tf.decode_raw(value, tf.uint8)
#
# USING tf.parse_single_example 
#  features = tf.parse_single_example(
#      value,
#      features={
#          'label': tf.FixedLenFeature([1], dtype=tf.string),
#          'image': tf.FixedLenFeature([48*48], dtype=tf.string),
#          'lx': tf.FixedLenFeature(shape=[144*4], dtype=tf.string)
#      })
#  result.label = tf.decode_raw(features['label'], tf.int32)
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]), 
		[result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  

  lx_uint8 = tf.strided_slice(record_bytes, [label_bytes+image_bytes], [label_bytes + image_bytes + lx_bytes])
  lx_int32 = tf.cast(lx_uint8, tf.int32)
  print('lx_int32=',lx_int32)
  lx_int32 = lx_int32 - 64
#  lx_ns = [ lx_int32[i] - 64  for i in range(0,144)] # compensation  -64
  
#  b=tf.string_split(value[image_bytes+1:image_bytes+1+lx_bytes],delimiter='').values
#
#  lx = tf.stack(lx_ns)
#  print('>> lx after stack =', lx)
#  lx.set_shape([144])
  lx = tf.reshape(lx_int32, [12,12])
  print('>> lx after reshape =', lx)

  lx  = tf.tile(lx,[2,2])
  result.landmark = tf.reshape(lx, [24,24,1])
  print('result.landmark after tile=',result.landmark)

  print('result=',result)
  return result


def _generate_image_and_label_batch(image, label, landmark, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch, landmarks = tf.train.shuffle_batch(
        [image, label, landmark],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch, landmarks = tf.train.batch(
        [image, label, landmark],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size]), landmarks


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.
  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # KSJHANG, gender_train
  # 48x48x3 : filenames = [os.path.join(data_dir+'gender_train/', '48x48x3_%d.bin' % i) for i in range(2)] # NUM_CLASSES=2 KSJHANG
  filenames = [os.path.join(data_dir, 'fer2013train_lxip.bin') ] 
  print('train files=',filenames)
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  with tf.name_scope('data_augmentation'):
    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    print('read_input ',read_input)

#    landmark = read_input.landmark
    landmark = tf.cast(read_input.landmark, tf.float32)
    # reshape 
    print('>> landmark =',landmark)

    height = IMAGE_SIZE
    width = IMAGE_SIZE
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

#    reshaped_image = tf.reshape(reshaped_image, [height, width, 1])

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    # fer2013 : 3 --> 1 for one channel data
    distorted_image = tf.random_crop(reshaped_image, [height, width, 1])
    print('>> distorted image =',distorted_image)

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)


    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.

    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # apply standardization for landmark
    landmark = tf.image.per_image_standardization(landmark)

    # Set the shapes of tensors.
    # fer2013 : 1 channel 
    float_image.set_shape([height, width, 1])
    read_input.label = tf.reshape(read_input.label, [1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label, landmark,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    # validation data  KSJHANG gender_val, gender_test KSJHANG
    filenames = [os.path.join(data_dir, 'fer2013train_lxip.bin')  ] 
    #filenames = [os.path.join(data_dir, 'fer2013pubtst_lxip.bin')  ] 
    print('filenames = ', filenames)
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    # test data  KSJHANG, gender_test
    # 48x48x4: filenames = [os.path.join(data_dir+'gender_test/', '48x48x3_%d.bin' % i) for i in range(2)] # NUM_CLASSES=2, KSJHANG for gender
    filenames = [os.path.join(data_dir, 'fer2013pritst_lxip.bin'),   os.path.join(data_dir, 'fer2013pubtst_lxip.bin')  ] 
    print('filenames = ', filenames)
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  with tf.name_scope('input'):
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    landmark = read_input.landmark
    # reshape 
    print('>> landmark before tile =',landmark)
    #landmark = tf.tile(landmark, [2,2])
    landmark = tf.cast(read_input.landmark, tf.float32)
    print('>> landmark after cast=',landmark)

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)


    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # apply standardization for landmark
    landmark = tf.image.per_image_standardization(landmark)

    # Set the shapes of tensors.
    # fer2013 - 1 channel
    float_image.set_shape([height, width, 1])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label, landmark,
                                         min_queue_examples, batch_size,
                                         shuffle=False)
