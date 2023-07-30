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
from mctr.modeling.custom_model import CustomModel
from tensorflow.keras.losses import binary_crossentropy

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
    "a_i_f": tf.io.VarLenFeature(tf.int64),
    "k_f": tf.io.FixedLenFeature([3], tf.float32),
    "k_seq_f": tf.io.VarLenFeature(tf.int64),
    "k_seq_fv": tf.io.VarLenFeature(tf.float32),
    "c_f": tf.io.FixedLenFeature([1], tf.int64)
}

# Parse features, using the above template.
def parse_record(record):
    parsed_example = tf.io.parse_single_example(record, features=feature_proto)
#`tf.io.VarLenFeature` 返回的是一个 `tf.SparseTensor` 对象，由于后面需要对序列特征进行
#padding操作，而pad函数只支持稠密张量，所以这里使用tf.sparse.to_dense将其转化为稠密张量
    features = {
        'u_f': parsed_example['u_f'],
        'u_c_f': tf.sparse.to_dense(parsed_example['u_c_f']), # 商品类目ID序列
        "u_c_fv": tf.sparse.to_dense(parsed_example['u_c_fv']), # 商品类目id序列权重

        'u_s_f': tf.sparse.to_dense(parsed_example['u_s_f']), # 商品店铺ID序列
        "u_s_fv": tf.sparse.to_dense(parsed_example['u_s_fv']), # 商品店铺ID序列权重

        'u_b_f': tf.sparse.to_dense(parsed_example['u_b_f']), # 商品品牌ID序列
        "u_b_fv": tf.sparse.to_dense(parsed_example['u_b_fv']), # 商品品牌ID序列权重

        'u_i_f': tf.sparse.to_dense(parsed_example['u_i_f']), # 商品意图ID序列
        "u_i_fv": tf.sparse.to_dense(parsed_example['u_i_fv']), # 商品意图ID序列权重

        "a_f": parsed_example['a_f'],
        "a_i_f":  tf.sparse.to_dense(parsed_example['a_i_f']), # 商品关联用户意图ID
        "k_f": parsed_example['k_f'],
        "k_seq_f": tf.sparse.to_dense(parsed_example['k_seq_f']), # 用户意图ID序列和商品关联用户意图ID组合特征
        "k_seq_fv": tf.sparse.to_dense(parsed_example['k_seq_fv']),
        "c_f": parsed_example['c_f']
    }

    y_label = parsed_example['y']
    print("y_label", y_label.shape)
    z_label = parsed_example['z']
    labels = {"y": y_label, "z": z_label}
    #return features, labels
    return features, [y_label, z_label]
# Apply the parsing to each record from the dataset.

def create_dataset(data, padded_shapes, padded_value, batch_size = 1000):
  data = data.map(parse_record)
  # Refill data indefinitely.
  data = data.repeat()
  # Shuffle data.
  data = data.shuffle(buffer_size=1000)
  # Batch data (aggregate records together).
  data = data.padded_batch(batch_size=batch_size, padded_shapes=padded_shapes)
  # Prefetch batch (pre-load batch for faster consumption).
  data = data.prefetch(buffer_size=1)
  return data

# 参考：
# https://towardsdatascience.com/building-your-first-neural-network-in-tensorflow-2-tensorflow-for-hackers-part-i-e1e2f1dfe7a0
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/5_DataManagement/tfrecords.ipynb
# https://github.com/Prayforhanluo/CTR_Algorithm/tree/main/tensorflow2.0
# https://github.com/linbang/DIN

def binary_crossentropy_cvr(y_true, y_pred, from_logits=False):
    y_ctr_true = y_true[:, 1]
    # y_cvr_true = y_true[:, 0:1]
    indices = tf.equal(y_ctr_true, 1)

    y_cvr_pred = y_pred[indices]
    y_cvr_true = y_cvr_true[indices]

    loss= binary_crossentropy(y_cvr_true, y_cvr_pred, from_logits=from_logits)
    return loss

def loss_func(y_true, y_pred):
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    # y_label = y_true['y']
    # z_label = y_true['z']
    click_label = y_true[0]
    conv_label = y_true[1]

    ctr_loss= binary_crossentropy(click_label, y_pred[0])
    ctcvr_loss= binary_crossentropy(conv_label, y_pred[1])
    loss = ctr_loss + ctcvr_loss
    return loss

def main(_):
    filenames = [FLAGS.tfrecord_input_dir + "/sample_skeleton_train_processed.parquet.tfrecord"]
    data = tf.data.TFRecordDataset(filenames)
    padded_shapes = (
        {
        'u_f' : [len(USER_Fields)],
        'u_c_f' : [None],
        'u_c_fv' : [None],

        'u_s_f' : [None],
        'u_s_fv' : [None],

        'u_b_f' : [None],
        'u_b_fv' : [None],

        'u_i_f' : [None],
        'u_i_fv' : [None],

        'a_f' : [4],

        'a_i_f' : [None],
        'k_f' : [len(CROSS_Fields)],
        'k_seq_f': [None],
        'k_seq_fv': [None],
        'c_f': [len(CONTEXT_Fields)]
        }, 
        # {
        # 'y': [1],
        # 'z': [1]
        # }
        [2, 1]
        )
    padded_value = (
        {
        'u_f' : 0,
        'u_c_f' : list(USER_CAT_SEQ_Fields.values())[0],
        'u_c_fv' : 0.0,

        'u_s_f' : list(USER_SHOP_SEQ_Fields.values())[0],
        'u_s_fv' : 0.0,

        'u_b_f' : list(USER_BRAND_SEQ_Fields.values())[0],
        'u_b_fv' : 0.0,

        'u_i_f' : list(USER_INTENTION_SEQ_Fields.values())[0],
        'u_i_fv' : 0.0,

        'a_f' : 0,

        'a_i_f' : list(AD_SEQ_Fields.values())[0],
        'k_f' : 0,
        'k_seq_f': list(CROSS_SEQ_Fields.values())[0],
        'k_seq_fv': 0.0,
        'c_f': 0
        }, 
        # {
        # 'y': 0,
        # 'z': 0 
        # }
        0
        )

    dataset = create_dataset(data, padded_shapes, padded_value)
    print("dataset", dataset)
    # 使用一个共用的embedding table, 每个特征各自有一个全局编码的默认值
    feature_dim, embed_dim, fc_dims, dropout = 10000000, 8, [128, 256], 0.5
    customModel = CustomModel(feature_dim, embed_dim, fc_dims, dropout)
    customModel.compile(optimizer='adam',
              loss=loss_func,
              metrics=['accuracy'])
    customModel.fit(dataset, epochs=5, steps_per_epoch=100)

    # i = 0
    # for record in dataset.take(100):
    #     print (type(record))
    #     feature = record[0]
    #     y = record[1]
    #     # z = record[2]
    #     a_f = feature['a_f']
    #     print(tf.shape(y))
    #     # print(y.numpy())
    #     print(tf.shape(a_f))
    #     # print(feature)
    #     i += 1
    #     print("u_s_f")
    #     print(i)
    #     print(feature['u_s_f'])

    #     print(feature['u_s_f'].shape)
    #     for i in range(feature['u_s_f'].shape[0]):
    #         print(i, len(feature['u_s_f'][i].numpy()), feature['u_s_f'][i].numpy())

    #     # print(record['y'].numpy())
    #     # print(record['z'].numpy())
        # print(record['fare'].numpy())

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)