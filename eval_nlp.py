#! /usr/bin/env python

from __future__ import absolute_import, division, print_function
from builtins import *

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
#import csv
import unicodecsv as csv
import codecs
import collections

# Parameters
# ==================================================

# Data Parameters

tf.flags.DEFINE_string("sentences_data_file_train", "./nlp_features/econ_health_sentences_list_of_strings.npy", "Data source for the training examples.")
tf.flags.DEFINE_string("labels_data_file_train", "./nlp_features/econ_health_onehotlabels.npy", "Data source for training labels.")
tf.flags.DEFINE_string("sentences_data_file_test", "./nlp_features/gop_sentences_list_of_strings.npy", "Data source for testing examples.")
tf.flags.DEFINE_string("labels_data_file_test", "./nlp_features/gop_onehotlabels.npy", "Data source for testing labels.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1488504630/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    #x_raw, y_test = FLAGS.positive_data_file, FLAGS.negative_data_file)
    #y_test = np.argmax(y_test, axis=1)
    x_raw = np.load(FLAGS.sentences_data_file_train)
    y_test = np.load(FLAGS.labels_data_file_train)
    y_test = np.argmax(y_test,axis=1)
else:
    #x_raw = ["a masterpiece four years in the making", "everything is off."]
    #y_test = [1, 0]
    x_ld = np.load(FLAGS.sentences_data_file_test)
    x_raw = [x.encode('utf-8') for x in x_ld] 
    y_test = np.load(FLAGS.labels_data_file_test)
    
    #handling econ health data
    if FLAGS.labels_data_file_test == "./nlp_features/econ_health_onehotlabels.npy":
        y = []
     #transforming labels to size 3 one-hots
        for lab in y_test:
            tmp = np.zeros(3)
            ii = int(np.nonzero(lab)[0])
            if 0<=ii and ii<=2:
                tmp[0] = 1
            elif 2<ii and ii<=5:
                tmp[1] = 1
            elif 5< ii and ii<=8:
                tmp[2] = 1
            y.append(tmp)

        y_test = np.array(y)

    y_test = np.argmax(y_test,axis=1)

#import pdb; pdb.set_trace()
print("Test set label balance:")
p = collections.Counter(y_test)
print(p)
# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'wb') as f:  
    #import pdb; pdb.set_trace() 
    csv.writer(f,encoding='utf-8').writerows(predictions_human_readable)
