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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys, getopt

import numpy as np
import tensorflow as tf

import model_landmark

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './emo_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
#tf.app.flags.DEFINE_string('checkpoint_dir', './cifar10_train_0.01_256_hsv',
tf.app.flags.DEFINE_string('checkpoint_dir', './emo_train',
                           """Directory where to read model checkpoints.""")
#tf.app.flags.DEFINE_integer('eval_interval_secs',  60*5,
tf.app.flags.DEFINE_integer('eval_interval_secs',  5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 20000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")


def eval_once(sess, saver, summary_writer, top_k_op, summary_op, global_step, coord):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """

  num_iter = int(math.ceil(num_examples / FLAGS.batch_size))
  true_count = 0  # Counts the number of correct predictions.
  total_sample_count = num_iter * FLAGS.batch_size
  step = 0
  print('num_iter=', num_iter)
  while (step < num_iter) :
    predictions, summary = sess.run([top_k_op,summary_op])
    true_count += np.sum(predictions)
    step += 1
    summary_writer.add_summary(summary, step)
      # Compute precision @ 1.
  precision = true_count / num_iter / FLAGS.batch_size
  print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
  return precision 

def evaluate(data_dir, eval_data):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    leval_data = eval_data == 'test'

    # data_dir = fold1, fold2, ... 
    images, labels, landmarks = model_landmark.inputs(eval_data=leval_data,data_dir=data_dir)
    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    # Build inference Graph.
    print('>>>>> input original size = ',images)
#   in case of images less than or larger than 227x227
#   images = tf.image.resize_images(images, [227,227] )
#   print('>>>>> input resized = ',images)
    if change_mode=='227':
      images = tf.image.resize_images(images, [227,227] )
      logits = model_landmark.inference_227(images, landmarks, 1.0)
    elif change_mode == 'nolx':
      logits = model_landmark.inference_nolx(images, 1.0)
    else:
      logits = model_landmark.inference(images, landmarks, 1.0)

    # Calculate accuracy.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    top_k_op = tf.cast(top_k_op, tf.int32)
    acc_batch = tf.reduce_sum(top_k_op)
    tf.summary.scalar(name="acc_batch", tensor=acc_batch/FLAGS.batch_size)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model_landmark.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(eval_dir, g)

    gpu_options = tf.GPUOptions(allocator_type='BFC',allow_growth=True)
    with tf.Session(config=tf.ConfigProto( allow_soft_placement=True,gpu_options=gpu_options) ) as sess:
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return

      coord = tf.train.Coordinator()
      # Start the queue runners.
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                       start=True))
      try:
        global no_runs
        sumprec = 0.0 
        for i in range(int(no_runs)): # 100 times repetition
          prec = eval_once(sess, saver, summary_writer, top_k_op, summary_op, global_step, coord )
          print('>> precision = ', prec)
          sumprec += prec
        print('>>  average precision = ', sumprec / no_runs)
      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)


def main(argv):  # pylint: disable=unused-argument
  global eval_dir, checkpoint_dir, num_examples, no_runs, eval_data, data_dir, change_mode
  # options
  try:
    opts, args = getopt.getopt(argv,"h:e:c:d:n:r:v:x:",["eval_dir=","checkpoint_dir=","no_examples=", "no_runs=" ] )
  except getopt.GetoptError:
    print( 'eval.py -c <checkpoint_dir> -e <eval_dir> -n <no_examples> -r <no_runs> -v <eval_data/test or else> ')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
       print( 'eval.py -c <checkpoint_dir> -e <eval_dir> -n <no_examples> -r <no_runs> -v <eval_data/test or else> ')
       sys.exit()
    elif opt in ("-e", "--eval_dir"):
       eval_dir = arg
    elif opt in ("-c", "--checkpoint_dir"):
       checkpoint_dir = arg
    elif opt in ("-d", "--data_dir"):
       data_dir = arg
    elif opt in ("-n", "--no_examples"):
       num_examples = float(arg)
    elif opt in ("-r", "--no_runs"):
       no_runs = float(arg)
    elif opt in ("-v", "--eval_data"):
       eval_data = arg
    elif opt in ("-x", "--change_mode"):
       change_mode = arg

  if tf.gfile.Exists(eval_dir):
    tf.gfile.DeleteRecursively(eval_dir)
  tf.gfile.MakeDirs(eval_dir)
  print('checkpoint_dir=', checkpoint_dir)
  print('data_dir=', data_dir)
  print('eval_dir=',eval_dir)
  print('no_examples=', num_examples)
  print('no_runs=', no_runs)
  print('eval_data=', eval_data)
  print('change_mode=', change_mode)

  evaluate(data_dir, eval_data)


eval_data = FLAGS.eval_data
eval_dir = FLAGS.eval_dir
checkpoint_dir=FLAGS.checkpoint_dir
num_examples=FLAGS.num_examples
no_runs= 20.0
data_dir = '../fer2013.bin/'
chane_mode = 'org'

if __name__ == '__main__':
  main(sys.argv[1:])

