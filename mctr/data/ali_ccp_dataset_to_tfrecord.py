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
USER_Fields = {'101' : 1, '121' : 2, '122' : 3, '124' : 4, '125' : 5, '126' : 6, '127' : 7, '128' : 8, '129' : 9}
USER_CAT_SEQ_Fields = {'109_14' : 10}      #user multi-hot feature
USER_SHOP_SEQ_Fields = {'110_14' : 11}      #user multi-hot feature
USER_BRAND_SEQ_Fields = {'127_14' : 12}      #user multi-hot feature
USER_INTENTION_SEQ_Fields = {'150_14' : 13}      #user multi-hot feature
# 需要记录每个序列的长度，后面padding需要用到
AD_Fields = {'205' : 14, '206' : 15, '207' : 16, '216' : 17}  #ad feature for DIN
AD_SEQ_Fields =  {'210' : 18} # 广告序列特征(用户意图序列)
# 需要记录该特征的长度，后面数据处理可能会用到
CROSS_Fields = {'508' : 19, '509' : 20, '702' : 21}
CROSS_SEQ_Fields = {'853' : 22} # 交叉特征中的序列特征
CONTEXT_Fields = {'301' : 23}

padded_shape_dict = {'109_14': 1001, '110_14': 1001, '127_14': 1001, '150_14': 335, '210': 38, '853': 25}

# 生成序列特征
# seq_fea_field: 序列特征对应的field名称
# feat_id_map: 存储特征field和特征id的map
# feat_val_map: 存储特征field和特征val的map
# padded_shape_dict 存储特征的最长长度的map
def gen_seq_fea(seq_fea_field, feat_id_map, feat_val_map):
    max_seq_len = padded_shape_dict[seq_fea_field]
    ad_seq_feat = [0] * max_seq_len
    ad_seq_feat_val = [0.0] * max_seq_len
    ad_seq_len_feat = [0]
    if seq_fea_field in feat_id_map:
        actual_seq_len = len(feat_id_map[seq_fea_field])
        ad_seq_feat[0 : actual_seq_len] = feat_id_map[seq_fea_field][:]
        ad_seq_feat = np.array(ad_seq_feat, dtype=np.int)
        ad_seq_feat_val[0 : actual_seq_len] = feat_val_map[seq_fea_field][:]
        ad_seq_feat_val = np.array(ad_seq_feat_val, dtype=np.float32)

        ad_seq_len_feat = [actual_seq_len]
        ad_seq_len_feat = np.array(ad_seq_len_feat, dtype=np.int)
    else:
        ad_seq_feat = np.array(ad_seq_feat, dtype=np.int)
        ad_seq_feat_val = np.array(ad_seq_feat_val, dtype=np.float32)
        ad_seq_len_feat = np.array(ad_seq_len_feat, dtype=np.int)

    return ad_seq_feat, ad_seq_feat_val, ad_seq_len_feat

def gen_seq_fea_no_padding(seq_fea_field_map, feat_id_map, feat_val_map):
    # max_seq_len = padded_shape_dict[seq_fea_field]
    seq_feat = [seq_fea_field_map.values()[0]]
    seq_feat_val = [0.0] # 用0做默认值
    seq_fea_field = seq_fea_field_map.keys()[0]
    # ad_seq_len_feat = [0]
    if seq_fea_field in feat_id_map:
        # actual_seq_len = len(feat_id_map[seq_fea_field])
        seq_feat = feat_id_map[seq_fea_field][:]
        seq_feat = np.array(seq_feat, dtype=np.int)
        seq_feat_val = feat_val_map[seq_fea_field][:]
        seq_feat_val = np.array(seq_feat_val, dtype=np.float32)

        # ad_seq_len_feat = [actual_seq_len]
        # ad_seq_len_feat = np.array(ad_seq_len_feat, dtype=np.int)
    else:
        seq_feat = np.array(seq_feat, dtype=np.int)
        seq_feat_val = np.array(seq_feat_val, dtype=np.float32)

    return seq_feat, seq_feat_val, []

def gen_tfrecords(in_file):
    basename = os.path.basename(in_file) + ".tfrecord"
    out_file = os.path.join(FLAGS.output_dir, basename)
    print("in_file %s, basename %s out_file %s"%(in_file, basename, out_file))
    tfrecord_out = tf.io.TFRecordWriter(out_file)
    parquet_file = pq.ParquetFile(in_file)
    num_lines = 0

    # print(parquet_file.metadata)
    # print(parquet_file.schema)
    for batch in parquet_file.iter_batches(batch_size = 1000):
        batch = batch.to_pandas()
        # print(batch.info(verbose = True))
        #https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
        for index, row in batch.iterrows():
            # print(index, row['sample_index'], row['click_label'], row['conv_label'],
            #       row['common_fea_index'], row['fea_num'], row['fea_str'])
            # print(row['click_label'])
            # print(type(row['click_label']))
            feature = {
                "y": tf.train.Feature(float_list = tf.train.FloatList(value=[row['click_label']])),
                "z": tf.train.Feature(float_list = tf.train.FloatList(value=[row['conv_label']]))
            }
            # 记录当前这个样本的每个特征field下面的特征id（序列）、特征值（序列）
            feat_id_map = {}
            feat_val_map = {}
            common_feat_str = row['common_fea_index']
            # todo: 下面可以抽象成函数
            for fstr in common_feat_str.split('\x01'):
                field, field_val = fstr.split('\x02') # field， feat_id, feat_val
                feat_id, feat_val = field_val.split('\x03')
                # feat_id = int(feat_id)
                # feat_val = float(feat_val)
                if field in feat_id_map:
                    # print(field, feat_id_map[field])
                    feat_id_map[field].append(feat_id)
                    feat_val_map[field].append(feat_val)
                else:
                    feat_id_map[field] = [feat_id]
                    feat_val_map[field] = [feat_val]

            fea_str = row['fea_str']
            for fstr in fea_str.split('\x01'):
                field, field_val = fstr.split('\x02') # field， feat_id, feat_val
                feat_id, feat_val = field_val.split('\x03')
                # feat_id = int(feat_id)
                # feat_val = float(feat_val)
                if field in feat_id_map:
                    feat_id_map[field].append(feat_id)
                    feat_val_map[field].append(feat_val)
                else:
                    feat_id_map[field] = [feat_id]
                    feat_val_map[field] = [feat_val]
            # 用户特征

            current_feat_field_num = 0
            user_feat= sorted(USER_Fields.values(), key= lambda x: x, reverse=False)
            # user_feat = [0] * len(USER_Fields) # 用0做默认值，但是如果文件中存在id为0的特征的话就不能这么做了。
            for field, fea_index in USER_Fields.items():
                # print("field %s fea_index %d"%(field, fea_index))
                if field in feat_id_map:
                    user_feat[fea_index - 1] = feat_id_map[field][0]
            # print("user_feat")
            # print(user_feat)
            user_feat = np.array(user_feat, dtype=np.int)
            # print(type(user_feat))
            feature.update({"u_f": tf.train.Feature(int64_list=tf.train.Int64List(value=user_feat))})

            user_cat_seq_feat, user_cat_seq_feat_val, _ = gen_seq_fea_no_padding(USER_CAT_SEQ_Fields, feat_id_map, feat_val_map)
            user_shop_seq_feat, user_shop_seq_feat_val, _ = gen_seq_fea_no_padding(USER_SHOP_SEQ_Fields, feat_id_map, feat_val_map)
            user_brand_seq_feat, user_brand_seq_feat_val, _ = gen_seq_fea_no_padding(USER_BRAND_SEQ_Fields, feat_id_map, feat_val_map)
            user_intention_seq_feat, user_intention_seq_feat_val, _ = gen_seq_fea_no_padding(USER_INTENTION_SEQ_Fields, feat_id_map, feat_val_map)
            feature.update({"u_c_f": tf.train.Feature(int64_list=tf.train.Int64List(value=user_cat_seq_feat)),
                            "u_c_fv": tf.train.Feature(float_list=tf.train.FloatList(value=user_cat_seq_feat_val))})

            feature.update({"u_s_f": tf.train.Feature(int64_list=tf.train.Int64List(value=user_shop_seq_feat)),
                            "u_s_fv": tf.train.Feature(float_list=tf.train.FloatList(value=user_shop_seq_feat_val))})

            feature.update({"u_b_f": tf.train.Feature(int64_list=tf.train.Int64List(value=user_brand_seq_feat)),
                            "u_b_fv": tf.train.Feature(float_list=tf.train.FloatList(value=user_brand_seq_feat_val))})

            feature.update({"u_i_f": tf.train.Feature(int64_list=tf.train.Int64List(value=user_intention_seq_feat)),
                            "u_i_fv": tf.train.Feature(float_list=tf.train.FloatList(value=user_intention_seq_feat_val))})

#            current_feat_field_num += len(USER_Fields)
            # 广告特征
            start_feat_index = min(AD_Fields.values())
            ad_feat = [0] * len(AD_Fields) # 也是用0做默认值
            for field, fea_index in AD_Fields.items():
                # print("field %s fea_index %d"%(field, fea_index))
                if field in feat_id_map:
                    ad_feat[fea_index - start_feat_index] = feat_id_map[field][0]
            ad_feat = np.array(ad_feat, dtype=np.int)
            print("ad_feat")
            print(ad_feat)
            print(np.shape(ad_feat))
            feature.update({"a_f": tf.train.Feature(int64_list=tf.train.Int64List(value=ad_feat))})

            # current_feat_field_num += len(AD_Fields)
            # 序列特征该咋处理
            # 应该先初始化为成固定长度，然后填充有值的那部分, 同时，记录这个序列的长度
            ad_seq_feat, _, _ = gen_seq_fea_no_padding(AD_SEQ_Fields, feat_id_map, feat_val_map)
            feature.update({"a_i_f": tf.train.Feature(int64_list=tf.train.Int64List(value=ad_seq_feat))})

            cross_feat = [0.0] * len(CROSS_Fields)
            start_feat_index = min(CROSS_Fields.values())
            for field, fea_index in CROSS_Fields.items():
                # print("field %s fea_index %d"%(field, fea_index))
                if field in feat_id_map:
                    cross_feat[fea_index - start_feat_index] = feat_id_map[field][0]
            cross_feat = np.array(cross_feat, dtype=np.float32)
            feature.update({"k_f": tf.train.Feature(float_list=tf.train.FloatList(value=cross_feat))})

            cross_seq_feat, cross_seq_fea_val, _ = gen_seq_fea_no_padding(CROSS_SEQ_Fields, feat_id_map, feat_val_map)
            feature.update({"k_seq_f": tf.train.Feature(int64_list=tf.train.Int64List(value=cross_seq_feat)),
                            "k_seq_fv": tf.train.Feature(float_list=tf.train.FloatList(value=cross_seq_fea_val))})

            start_feat_index = min(CONTEXT_Fields.values())
            context_feat = [0] * len(CONTEXT_Fields)
            for field, fea_index in CONTEXT_Fields.items():
                if field in feat_id_map:
                    context_feat[fea_index - start_feat_index] = feat_id_map[field][0]
            context_feat = np.array(context_feat, dtype = np.int)
            feature.update({"c_f": tf.train.Feature(int64_list=tf.train.Int64List(value=context_feat))})

            example = tf.train.Example(features = tf.train.Features(feature = feature))
            serialized = example.SerializeToString()
            tfrecord_out.write(serialized)
            num_lines += 1
        print("Process %d" % num_lines)
        if num_lines % 10000 == 0:
            print("Process %d" % num_lines)
            break
        # break
    tfrecord_out.close()

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