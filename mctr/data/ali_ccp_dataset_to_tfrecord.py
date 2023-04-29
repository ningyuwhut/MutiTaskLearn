#!/usr/bin/env python
#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 基于 https://github.com/lambdaji/tf_repos/blob/master/deep_ctr/Feature_pipeline/get_aliccp_tfrecord.py
import sys
import os
import glob

import tensorflow as tf
import numpy as np
import re
from multiprocessing import Pool as ThreadPool
# import fastparquet as fp
import pyarrow as pa
import pyarrow.parquet as pq

# 切换到tensorflow 2.0
# https://github.com/tensorflow/tensorflow/issues/34431
# https://github.com/abseil/abseil-py/blob/main/smoke_tests/sample_app.py

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

# flags = tf.app.flags
FLAGS = flags.FLAGS
# LOG = tf.logging

ALI_CCP_DATASET_DIR = "/home/data/dataset/ali_ccp"
flags.DEFINE_string("input_dir", ALI_CCP_DATASET_DIR + "/" + "parquet", "input dir")
flags.DEFINE_string("output_dir", ALI_CCP_DATASET_DIR + "/" + "tfrecord", "output dir")
flags.DEFINE_integer("threads", 1, "threads num")

#保证顺序以及字段数量
#User_Fields = set(['101','109_14','110_14','127_14','150_14','121','122','124','125','126','127','128','129'])
#Ad_Fields = set(['205','206','207','210','216'])
#Cross_Fields = set(['508','509','702','853'])
#Context_Fields = set(['301'])
DISCRETE_Fields = {'101' : '1', '121' : '2', '122' : '3', '124' : '4', '125' : '5', '126' : '6', '127' : '7', '128' : '8', '129' : '9', 
                   '301' : '10'}
USER_SEQUENCE_Fields = {'109_14' : ('u_cat', '11'), '110_14' : ('u_shop' , '12'), '127_14' : ('u_brand' , '13'), '150_14' : ('u_int' , '14')}      #user multi-hot feature
Ad_Fields = {'205' : '15', '206' : ('a_cat' , '16'), '207' : ('a_shop', '17'), '210' : ('a_int', '18'),'216' : ('a_brand', '19')}  #ad feature for DIN
CROSS_Fileds = {'508' : '20', '509' : '21', '702' : '22', '853' : '23'}

#40362692,0,0,216:9342395:1.0 301:9351665:1.0 205:7702673:1.0 206:8317829:1.0 207:8967741:1.0 508:9356012:2.30259 210:9059239:1.0 210:9042796:1.0 210:9076972:1.0 210:9103884:1.0 210:9063064:1.0 127_14:3529789:2.3979 127_14:3806412:2.70805

def gen_tfrecords(in_file):
    basename = os.path.basename(in_file) + ".tfrecord"
    out_file = os.path.join(FLAGS.output_dir, basename)
    print("in_file %s, basename %s out_file %s"%(in_file, basename, out_file))
    # tfrecord_out = tf.python_io.TFRecordWriter(out_file)
    parquet_file = pq.ParquetFile(in_file)

    print(parquet_file.metadata)
    print(parquet_file.schema)
    for batch in parquet_file.iter_batches(batch_size = 1000):
        print(type(batch))
        batch = batch.to_pandas() 
        print(type(batch))
        print(batch.info(verbose = True))

        #https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
        for index, row in batch.iterrows():
    #        print(index, row['sample_index'], row['click_label'], row['conv_label'],
    #              row['common_fea_index'], row['fea_num'], row['fea_str'])

            feat_id_map = {}
            feat_val_map = {}
            common_feat_str = row['common_fea_index']
            fea_str = row['fea_str']
            # todo: 下面可以抽象成函数
            for fstr in common_feat_str.split('\x01'):
                field, feat_val = fstr.split('\x02') # field， feat_id, feat_val
                feat_id, feat_val = feat_val.split('\x03')
                if field in feat_id_map:
                    feat_id_map[field] = feat_id_map[field].append(feat_id)
                    feat_val_map[field] = feat_val_map[field].append(feat_val)
                else:
                    feat_id_map[field] = [feat_id]
                    feat_val_map[field] = [feat_val]
            

            for fstr in fea_str.split('\x01'):
                field, feat_val = fstr.split('\x02') # field， feat_id, feat_val
                feat_id, feat_val = feat_val.split('\x03')
                if field in feat_id_map:
                    feat_id_map[field] = feat_id_map[field].append(feat_id)
                    feat_val_map[field] = feat_val_map[field].append(feat_val)
                else:
                    feat_id_map[field] = [feat_id]
                    feat_val_map[field] = [feat_val]
            for field in DISCRETE_Fields:

            for field in USER_SEQUENCE_Fields:
            for field in Ad_Fields:
            for field in CROSS_Fileds:

                

    # for df in pf.iter_row_groups():
        # for line in fi:
        #     fields = line.strip().split(',')
        #     if len(fields) != 4:
        #         continue
        #     #1 label
        #     y = [float(fields[1])]
        #     z = [float(fields[2])]
        #     feature = {
        #         "y": tf.train.Feature(float_list = tf.train.FloatList(value=y)),
        #         "z": tf.train.Feature(float_list = tf.train.FloatList(value=z))
        #      }

        #     splits = re.split('[ :]', fields[3])
        #     ffv = np.reshape(splits,(-1,3))
        #     #common_mask = np.array([v in Common_Fileds for v in ffv[:,0]])
        #     #af_mask = np.array([v in Ad_Fileds for v in ffv[:,0]])
        #     #cf_mask = np.array([v in Context_Fileds for v in ffv[:,0]])

        #     #2 不需要特殊处理的特征
        #     feat_ids = np.array([])
        #     #feat_vals = np.array([])
        #     for f, def_id in Common_Fileds.iteritems():
        #         if f in ffv[:,0]:
        #             mask = np.array(f == ffv[:,0])
        #             feat_ids = np.append(feat_ids, ffv[mask,1])
        #             #np.append(feat_vals,ffv[mask,2].astype(np.float))
        #         else:
        #             feat_ids = np.append(feat_ids, def_id)
        #             #np.append(feat_vals,1.0)
        #     feature.update({"feat_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=feat_ids.astype(np.int)))})
        #                     #"feat_vals": tf.train.Feature(float_list=tf.train.FloatList(value=feat_vals))})

        #     #3 特殊字段单独处理
        #     for f, (fname, def_id) in UMH_Fileds.iteritems():
        #         if f in ffv[:,0]:
        #             mask = np.array(f == ffv[:,0])
        #             feat_ids = ffv[mask,1]
        #             feat_vals= ffv[mask,2]
        #         else:
        #             feat_ids = np.array([def_id])
        #             feat_vals = np.array([1.0])
        #         feature.update({fname+"ids": tf.train.Feature(int64_list=tf.train.Int64List(value=feat_ids.astype(np.int))),
        #                         fname+"vals": tf.train.Feature(float_list=tf.train.FloatList(value=feat_vals.astype(np.float)))})

        #     for f, (fname, def_id) in Ad_Fileds.iteritems():
        #         if f in ffv[:,0]:
        #             mask = np.array(f == ffv[:,0])
        #             feat_ids = ffv[mask,1]
        #         else:
        #             feat_ids = np.array([def_id])
        #         feature.update({fname+"ids": tf.train.Feature(int64_list=tf.train.Int64List(value=feat_ids.astype(np.int)))})

            # serialized to Example
    #         example = tf.train.Example(features = tf.train.Features(feature = feature))
    #         serialized = example.SerializeToString()
    #         tfrecord_out.write(serialized)
    #         #num_lines += 1
    #         #if num_lines % 10000 == 0:
    #         #    print("Process %d" % num_lines)
    # tfrecord_out.close()

def main(_):
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)
    path_pattern = os.path.join(FLAGS.input_dir, "*")
    print(path_pattern)
    file_list = glob.glob(path_pattern)
    print("total files: %d" % len(file_list))

    pool = ThreadPool(FLAGS.threads) # Sets the pool size
    pool.map(gen_tfrecords, file_list)
    pool.close()
    pool.join()

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)