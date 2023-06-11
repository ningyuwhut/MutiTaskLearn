
# from ali_ccp_dataset_to_tfrecord import *
from absl import app
from absl import flags
from absl import logging

import os
import sys
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from mctr.data.ali_ccp_dataset_to_tfrecord import *

FLAGS = flags.FLAGS
flags.DEFINE_string("tfrecord_input_dir", ALI_CCP_DATASET_DIR + "/" + "tfrecord", "input dir")
# Build features template, with types.
feature_proto = {
    'y':  tf.io.FixedLenFeature([1], tf.float32),
    'z':  tf.io.FixedLenFeature([1], tf.float32),
    'u_f':  tf.io.FixedLenFeature([9], tf.int64),
    'u_c_f':  tf.io.VarLenFeature(tf.int64),
    "u_c_fv": tf.io.VarLenFeature(tf.float32),
    'u_s_f':  tf.io.VarLenFeature(tf.int64),
    "u_s_fv": tf.io.VarLenFeature(tf.float32),
    'u_b_f':  tf.io.VarLenFeature(tf.int64),
    "u_b_fv": tf.io.VarLenFeature(tf.float32),

    'u_i_f':  tf.io.VarLenFeature(tf.int64),
    "u_i_fv": tf.io.VarLenFeature(tf.float32),
    "a_f": tf.io.FixedLenFeature([4], tf.int64),
    "a_i_f": tf.io.VarLenFeature( tf.int64),
    "k_f": tf.io.FixedLenFeature([3], tf.float32),
    "k_seq_f": tf.io.VarLenFeature(tf.int64),
    "k_seq_fv": tf.io.VarLenFeature(tf.float32),
    "c_f": tf.io.FixedLenFeature([1], tf.int64)
}


# Parse features, using the above template.
def parse_record(record):
    parsed_example = tf.io.parse_single_example(record, features=feature_proto)
    features = {
        'u_f': parsed_example['u_f'],
        'u_c_f': parsed_example['u_c_f'],
        "u_c_fv": parsed_example['u_c_fv'],
        'u_s_f': parsed_example['u_s_f'],
        "u_s_fv": parsed_example['u_s_fv'],
        'u_b_f':  parsed_example['u_b_f'],
        "u_b_fv": parsed_example['u_b_fv'],
        'u_i_f':  parsed_example['u_i_f'],
        "u_i_fv": parsed_example['u_i_fv'],
        "a_f": parsed_example['a_f'],
        "a_i_f": parsed_example['a_i_f'],
        "k_f": parsed_example['k_f'],
        "k_seq_f": parsed_example['k_seq_f'],
        "k_seq_fv": parsed_example['k_seq_fv'],
        "c_f": parsed_example['c_f']
    }
    y_label = parsed_example['y']
    z_label = parsed_example['z']
    return features, y_label, z_label
# Apply the parsing to each record from the dataset.

# Load TFRecord data.
def create_dataset(data):
  # ys = tf.one_hot(ys, depth=n_classes)
  # return tf.data.Dataset.from_tensor_slices((xs, ys)) \
  #   .map(preprocess) \
  #   .shuffle(len(ys)) \
  #   .batch(128)
  data = data.map(parse_record)
  # Refill data indefinitely.
  data = data.repeat()
  # Shuffle data.
  data = data.shuffle(buffer_size=1000)
  # Batch data (aggregate records together).
  data = data.batch(batch_size=256)
  # Prefetch batch (pre-load batch for faster consumption).
  data = data.prefetch(buffer_size=1)
  return data

# ====================================
# # Define the input data shape
# feature_description = {
#     'feature1': tf.io.FixedLenFeature([], tf.float32),
#     'feature2': tf.io.FixedLenFeature([], tf.int64),
#     'label': tf.io.FixedLenFeature([], tf.int64),
# }
# # Define a function to parse each example in the TFRecord dataset
# def _parse_function(example_proto):
#     parsed_example = tf.io.parse_single_example(example_proto, feature_description)
#     features = {
#         'feature1': parsed_example['feature1'],
#         'feature2': parsed_example['feature2']
#     }
#     label = parsed_example['label']
#     return features, label
# # Load the TFRecord dataset
# dataset = tf.data.TFRecordDataset('data.tfrecord').map(_parse_function)
# Define the DNN model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
# # Compile the model
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(),
#               metrics=['accuracy'])
# # Train the model
# model.fit(dataset.repeat(), epochs=10)
# # Evaluate the model
# loss, accuracy = model.evaluate(dataset.batch(32))
# print('Loss: {}, Accuracy: {}'.format(loss, accuracy))

# 参考：
# https://towardsdatascience.com/building-your-first-neural-network-in-tensorflow-2-tensorflow-for-hackers-part-i-e1e2f1dfe7a0
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/5_DataManagement/tfrecords.ipynb
# https://github.com/Prayforhanluo/CTR_Algorithm/tree/main/tensorflow2.0
# https://github.com/linbang/DIN

def main(_):
    filenames = [FLAGS.tfrecord_input_dir + "/sample_skeleton_train_processed.parquet.tfrecord"]
    data = tf.data.TFRecordDataset(filenames)
    dataset = create_dataset(data)

    for record in dataset.take(1):
        print (type(record))
        feature = record[0]
        y = record[1]
        z = record[2]
        a_f = feature['a_f']
        print(tf.shape(y))
        # print(y.numpy())
        print(tf.shape(a_f))
        # print(feature)
        print("k_seq_f")
        print(feature['k_seq_f'])

        # print(record['y'].numpy())
        # print(record['z'].numpy())
        # print(record['fare'].numpy())

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)