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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)

#from tensorflow.python import debug as tf_debug

from datetime import datetime

import os.path
import re
import time
import sys, getopt

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import model_landmark

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './landmark_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 6,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#tf.app.flags.DEFINE_integer('batch_size', 48,
#                            """Number of images to process in a batch.""")


def tower_loss(scope, images, labels, landmarks):
  """Calculate the total loss on a single tower running the CIFAR model.
  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].
  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  print('>>>>> input original = ',images)
#KSJAHNG, 227 mode
  #print('>>>>> input resized = ',images)
  global change_mode, dropout
  if change_mode == '227':
    images = tf.image.resize_images(images, [227,227] )
    logits = model_landmark.inference_227(images, landmarks, dropout)
  elif change_mode == 'nolx':
    logits = model_landmark.inference_nolx(images, dropout)
  elif change_mode == 'nolx227':
    images = tf.image.resize_images(images, [227,227] )
    logits = model_landmark.inference_nolx227(images, dropout)
  else:
    logits = model_landmark.inference(images, landmarks, dropout)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = model_landmark.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # to calculate accuracy for batch 
  top_k_op = tf.nn.in_top_k(logits, labels, 1)
  top_k_op = tf.cast(top_k_op, tf.int32)
  acc_batch = tf.reduce_sum(top_k_op)
  tf.summary.scalar(name="acc_batch", tensor=acc_batch/FLAGS.batch_size)

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % model_landmark.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)

  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (no_examples_epoch / FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * no_epochs_decay )
    print('>> decay_steps = ', decay_steps)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_lr, global_step, decay_steps, decay_factor, staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr)

    # Get images and labels for CIFAR-10.
    global data_dir

    images, labels, landmarks = model_landmark.distorted_inputs(data_dir)
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [images, labels, landmarks], capacity=2 * no_gpus)
    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(no_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (model_landmark.TOWER_NAME, i)) as scope:
            # Dequeues one batch for the GPU
            image_batch, label_batch, lx_batch = batch_queue.dequeue()
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss = tower_loss(scope, image_batch, label_batch, lx_batch)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        model_landmark.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4,allocator_type='BFC',allow_growth=True)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))
#    sess = tf.InteractiveSession()
    sess.run(init)
#    sess = tf_debug.LocalCLIDebugWrapperSession(sess)#, ui_type=FLAGS.ui_type)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    for step in xrange(max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time
#      print('lx_batch=',lx_batch)
#      print('loss_value =', loss_value)
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
#tf.get_collection(tf.GraphKeys.VARIABLES)
#tf.all_variables()
      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * no_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / no_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == max_steps:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv):  # pylint: disable=unused-argument
  # globals variables configurable with arguments 
  #cifar10.maybe_download_and_extract()
  global train_dir, decay_factor, initial_lr, dropout, max_steps, no_epochs_decay, no_gpus, no_examples_epoch, data_dir, change_mode
  if tf.gfile.Exists(train_dir):
    tf.gfile.DeleteRecursively(train_dir)
  tf.gfile.MakeDirs(train_dir)
  try:
    opts, args = getopt.getopt(argv,"h:i:f:d:e:t:m:g:n:x:c:",["initial=","factor=","dropout=","epochs=","train_dir=", "max_steps=", "no_gpus=", "no_examples","no_gpus=","example_dir=","change_mode="])
  except getopt.GetoptError:
    print( 'train.py -i <initial_lr> -d <dropout ratio> -e <no of epochs for decay> -t <train_dir> -m <max steps> -g <# of gpus> -f <decay factor> -n <no_examples_per_epoch> -x <example_dir> -c <change_mode> ')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
       print( 'train.py -i <initial_lr> -d <dropout ratio> -e <no of epochs for decay> -t <train_dir> -m <max steps> -g <# of gpus> -f <decay factor> -n <no_examples_per_epoch> -x <example_dir> -c <change_mode> ')
       sys.exit()
    elif opt in ("-i", "--initial"):
       initial_lr = float(arg)
    elif opt in ("-f", "--factor"):
       decay_factor = float(arg)
    elif opt in ("-d", "--dropout"):
       dropout = float(arg)
    elif opt in ("-e", "--epochs"):
       no_epochs_decay = float(arg)
    elif opt in ("-t", "--train_dir"):
       train_dir = arg
    elif opt in ("-m", "--max_steps"):
       max_steps = int(arg)
    elif opt in ("-g", "--no_gpus"):
       no_gpus = int(arg)
    elif opt in ("-n", "--no_examples"):
       no_examples_epoch = float(arg)
    elif opt in ("-x", "--example_dir"):
       data_dir = arg
    elif opt in ("-c", "--change_mode"):
       change_mode = arg 

  print('initial_lr=',initial_lr)
  print('decay_factor=',decay_factor)
  print('dropout=',dropout)
  print('no_epochs_decay=',no_epochs_decay)
  print('train_dir=',train_dir)
  print('max_steps=',max_steps)
  print('no_gpus=',no_gpus)
  print('no_examples_epoc=',no_examples_epoch)
  print('data_dir = ', data_dir)
  print('change_mode = ', change_mode)
  train()


initial_lr = model_landmark.INITIAL_LEARNING_RATE
decay_factor = model_landmark.LEARNING_RATE_DECAY_FACTOR
no_epochs_decay = model_landmark.NUM_EPOCHS_PER_DECAY
no_examples_epoch = model_landmark.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN 
dropout = 0.5
max_steps = FLAGS.max_steps
no_gpus = FLAGS.num_gpus
train_dir = FLAGS.train_dir
data_dir = "../fer2013.bin/"
change_mode = 'org'
if __name__ == '__main__':
  main(sys.argv[1:])

