#encoding=utf-8

from memory_profiler import profile
import time
from functools import partial
import pandas as pd
import sys
import os

ALI_CCP_DATASET_DIR = "/home/data/dataset/ali_ccp"
COMMON_FEATURE_TRAIN_SUBSET_FILE = ALI_CCP_DATASET_DIR + "/" + "common_features_train_sample.csv"
COMMON_FEATURE_TRAIN_FILE = ALI_CCP_DATASET_DIR + "/" + "common_features_train.csv"
TRAIN_SAMPLE_SUBSET_FILE = ALI_CCP_DATASET_DIR + "/" + "sample_skeleton_train_sample.csv"
TRAIN_SAMPLE_FILE = ALI_CCP_DATASET_DIR + "/" + "sample_skeleton_train.csv"

@profile
def process_common_feature(common_feature_file):
    common_feature_map = {}
    with open(common_feature_file, 'r') as f:
        for line in f:
            splits = line.strip().split(',')
            split_len = len(splits)
            if split_len != 3:
                print('%s have %d field, not 3'%(line, split_len))
                continue
            #common_feature_index|feat_num|feat_list
            common_feature_index = splits[0]
            feat_num = int(splits[1])
            feat_strs = splits[2]
            if len(feat_strs.split('\x01')) != feat_num:
                print('%s doesn\'t have %d feature'%(common_feature_index, feat_num))
                continue
            feat_map = {}
            for fstr in feat_strs.split('\x01'):
                field, feat_val = fstr.split('\x02')
                # feat, val = feat_val.split('\x03')
                # feat_lists.append('%s:%s:%s' % (filed,feat,val))
                if field in feat_map:
                    feat_map[field] = feat_map[field] + "\x02" + feat_val
                else:
                    feat_map[field] = feat_val
            common_feature_map[common_feature_index] = feat_map
    return common_feature_map

# 参考https://stackoverflow.com/questions/23765360/processing-large-files-in-python-1000-gb-or-more
# 这个方案有一个问题，就是读取出来之后的line为什么没有换行符呢
# @profile
def process_common_feature_v2(common_feature_file, size_in_bytes):
    common_feature_map = {}
    with open(common_feature_file, 'rb') as f:
        prev = ''
        count = 0
        f_read  = partial(f.read, size_in_bytes)
        for line in iter(f_read, ''):
            line = line.decode()
            # print(line)
            if count > 0 and count % 10000 == 0:
                print("processed %d lines"%(count))
            if (not line.endswith('\n')) and '\n' in line:
                text, rest = line.rsplit('\n', 1)
                text =  prev + text
                prev = rest
            elif line.endswith('\n'):
                text =  prev + line
                prev = ''
            else:
                if "\n" not in line:
                    print("not expected")
                if '\r' in line:
                    print('\r in line')
                if '\r\n' in line:
                    print('\r\n in line')
                print('line count %d hit else branch'%(count))
                count +=1
                # print(line)
                # break 

            splits_lines = text.strip().split('\n') #.split(',')
            for splits_line in splits_lines:
                splits = splits_line.strip().split(',')
                split_len = len(splits)
                if split_len != 3:
                    print('line count %d has %d field, not 3'%(count, split_len))
                    count += 1
                    continue
                #common_feature_index|feat_num|feat_list
                common_feature_index = splits[0]
                feat_num = int(splits[1])
                feat_strs = splits[2]
                if len(feat_strs.split('\x01')) != feat_num:
                    print('%s doesn\'t have %d feature'%(common_feature_index, feat_num))
                    count += 1
                    continue
                feat_map = {}
                for fstr in feat_strs.split('\x01'):
                    field, feat_val = fstr.split('\x02')
                    # feat, val = feat_val.split('\x03')
                    # feat_lists.append('%s:%s:%s' % (filed,feat,val))
                    if field in feat_map:
                        feat_map[field] = feat_map[field] + "\x02" + feat_val
                    else:
                        feat_map[field] = feat_val
                common_feature_map[common_feature_index] = feat_map
                count += 1
                # break
    return common_feature_map

@profile
def process_common_feature_v3(common_feature_file, chunksize = 10000):
    common_feature_map = {}
    with pd.read_csv(common_feature_file, chunksize = chunksize, iterator = True,
        names = ['common_fea_index', 'fea_num', 'fea_str']) as reader:
        count = 0
        for chunk in reader:
            print(type(chunk))  # 这是一个dataframe
            # todo 过滤掉 fea_num 和fea_str中的特征数量不同的行
            chunk_common_feature_map = dict(zip(chunk['common_fea_index'], chunk['fea_str']))
            common_feature_map.update(chunk_common_feature_map) 
            print(count)
            count +=1
            # for line in chunk:
            #     splits = line.strip().split(',')
            #     split_len = len(splits)
            #     if split_len != 3:
            #         print('line count %d has %d field, not 3'%(count, split_len))
            #         count += 1
            #         print(line)
            #         sys.exit() 
            #     #common_feature_index|feat_num|feat_list
            #     common_feature_index = splits[0]
            #     feat_num = int(splits[1])
            #     feat_strs = splits[2]
            #     if len(feat_strs.split('\x01')) != feat_num:
            #         print('%s doesn\'t have %d feature'%(common_feature_index, feat_num))
            #         count += 1
            #         continue
            #     feat_map = {}
            #     for fstr in feat_strs.split('\x01'):
            #         field, feat_val = fstr.split('\x02')
            #         # feat, val = feat_val.split('\x03')
            #         # feat_lists.append('%s:%s:%s' % (filed,feat,val))
            #         if field in feat_map:
            #             feat_map[field] = feat_map[field] + "\x02" + feat_val
            #         else:
            #             feat_map[field] = feat_val
            #     common_feature_map[common_feature_index] = feat_map
            #     count += 1
            # break
    print("common_feature_map.size: %d"%(len(common_feature_map)))
    return common_feature_map


@profile
def process_sample_file(sample_input_file, sample_output_file, common_feature_map):
    count = 0
    with open(sample_input_file, 'r') as f_input, open(sample_output_file, 'w') as f_output:
        for line in f_input:
            splits = line.strip().split(',')
            split_len = len(splits)
            if split_len != 6:
                print('%s doesn\'t only have %d field'%(line, split_len))
                count +=1
                continue
            feat_lists = []
            if(splits[1] == '0' and splits[2] == '1'):
                count +=1
                continue
            if count > 0 and count % 10000 == 0:
                print('line count %d'%(count))
            #sample_id|y|z|common_feature_index|feat_num|feat_list
            sample_id = splits[0]
            click_label = splits[1]
            conv_label = splits[2]
            common_feature_index = splits[3]
            feat_num = splits[4]
            feat_strs = splits[5]
            if common_feature_index in common_feature_map:
                feat_map = common_feature_map[common_feature_index]
                feat_map_str = '\x01'.join(feat_map)
                f_output.write("%s,%s,%s,%s,%s,%s"%(sample_id, click_label, conv_label,
                feat_map_str, feat_num, feat_strs))
            else:
                print('sample %s doesn\'t only have common_feature'%(sample_id))
                f_output.write(line)
            count += 1

# 参考https://stackoverflow.com/questions/20250771/remap-values-in-pandas-column-with-a-dict-preserve-nans
@profile
def process_sample_file_v2(sample_input_file, sample_output_file, common_feature_map, chunksize = 500000):
    count = 0
    with pd.read_csv(sample_input_file, chunksize = chunksize, iterator = True,
        names = ['sample_index', 'click_label', 'conv_label', 'common_fea_index',
                 'fea_num', 'fea_str']) as reader:
        for chunk in reader:
            print(type(chunk))
            # chunk_common_feature_map = dict(zip(chunk.common_index, chunk.fea_str))
            # common_feature_map.update(chunk_common_feature_map) 
            chunk['common_fea_index'] = chunk['common_fea_index'].map(common_feature_map).fillna(chunk['common_fea_index'])
            print(count)
            count += 1
            # https://stackoverflow.com/questions/74142416/appending-the-parquet-file-while-chunking
            if not os.path.isfile(sample_output_file):
                chunk.to_parquet(sample_output_file, engine='fastparquet')
            else:
                chunk.to_parquet(sample_output_file, engine='fastparquet', append = True)

        # for line in f_input:
        #     splits = line.strip().split(',')
        #     split_len = len(splits)
        #     if split_len != 6:
        #         print('%s doesn\'t only have %d field'%(line, split_len))
        #         count +=1
        #         continue
        #     feat_lists = []
        #     if(splits[1] == '0' and splits[2] == '1'):
        #         count +=1
        #         continue
        #     if count > 0 and count % 10000 == 0:
        #         print('line count %d'%(count))
        #     #sample_id|y|z|common_feature_index|feat_num|feat_list
        #     sample_id = splits[0]
        #     click_label = splits[1]
        #     conv_label = splits[2]
        #     common_feature_index = splits[3]
        #     feat_num = splits[4]
        #     feat_strs = splits[5]
        #     if common_feature_index in common_feature_map:
        #         feat_map = common_feature_map[common_feature_index]
        #         feat_map_str = '\x01'.join(feat_map)
        #         f_output.write("%s,%s,%s,%s,%s,%s"%(sample_id, click_label, conv_label,
        #         feat_map_str, feat_num, feat_strs))
        #     else:
        #         print('sample %s doesn\'t only have common_feature'%(sample_id))
        #         f_output.write(line)
        #     count += 1

if __name__ == "__main__":
    sample_output_file = ALI_CCP_DATASET_DIR + "/" + "sample_skeleton_train_processed.parquet"
    start = time.time()
    # process_common_feature is too slow to read a 10G file
    # common_feat_map = process_common_feature(COMMON_FEATURE_TRAIN_FILE)
    size_in_bytes = 1024*1024  #1073741824 #1次1G
    #common_feat_map = process_common_feature_v2(COMMON_FEATURE_TRAIN_FILE, size_in_bytes)
    common_feat_map = process_common_feature_v3(COMMON_FEATURE_TRAIN_FILE)
    end = time.time()
    print("process_common_feature cost %d min"%((end - start)/60))
    process_sample_file_v2(TRAIN_SAMPLE_FILE, sample_output_file, common_feat_map)
    end2 = time.time()
    # print("process_sample_file cost %d min"%((end2 - end)/60))



