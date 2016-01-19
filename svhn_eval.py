# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Evaluation for CIFAR-10.

Accuracy:
svhn_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by svhn_eval.py.

Speed:
On a single Tesla K40, svhn_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

import svhn
import svhn_input
import os
import svhn_flags
from PIL import Image

FLAGS = tf.app.flags.FLAGS

def inputs():
  filenames = [os.path.join(FLAGS.test_dir, FLAGS.test_file)]
  for f in filenames:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  filename_queue = tf.train.string_input_producer(filenames,shuffle=False)
  read_input = svhn_input.read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  height = FLAGS.image_size
  width = FLAGS.image_size
  float_image = tf.image.per_image_whitening(reshaped_image)
  num_preprocess_threads = 1
  images, label_batch = tf.train.batch(
      [float_image, read_input.label],
      batch_size=FLAGS.batch_size,
      num_threads=num_preprocess_threads,
      capacity=FLAGS.batch_size)
  tf.image_summary('images', images, max_images = 29)
  return images, tf.reshape(label_batch, [FLAGS.batch_size])

def eval_once(saver, summary_writer, top_k_op, top_k_predict_op, summary_op, images):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      print ("Checkpoint File:",ckpt.model_checkpoint_path)
      print ("Test Dir:",FLAGS.test_dir)
      print ("Test File:", FLAGS.test_file)
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/svhn_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      print (num_iter)
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        image, test_labels = sess.run([images,top_k_predict_op])
        # im = Image.fromarray(np.array(image).reshape(50,50,1).astype(np.uint8))
        print (step, int(test_labels[0]))
        # print (FLAGS.predictions_dir + "/" + str(step) + "_" + str(int(test_labels[0])) + ".jpg")
        # im.save("tmp/svhn_results/"+str(step) + "_" + str(int(test_labels[0])) + ".jpg")
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print (total_sample_count)
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels for CIFAR-10.
    images, labels = inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = svhn.inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    top_k_predict_op = tf.argmax(logits,1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        svhn.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    while True:
      eval_once(saver, summary_writer, top_k_op, top_k_predict_op, summary_op, images)
      break


def main(argv=None):  # pylint: disable=unused-argument
  if gfile.Exists(FLAGS.eval_dir):
    gfile.DeleteRecursively(FLAGS.eval_dir)
  if gfile.Exists(FLAGS.predictions_dir):
    gfile.DeleteRecursively(FLAGS.predictions_dir)
  gfile.MakeDirs(FLAGS.eval_dir)
  gfile.MakeDirs(FLAGS.predictions_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
