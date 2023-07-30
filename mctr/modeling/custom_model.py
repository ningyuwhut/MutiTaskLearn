

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, Dropout
from tensorflow.keras import Model
from mctr.modeling.din import DeepInterestNet
from mctr.modeling.my_mmoe import MyMMOE
from mctr.data.ali_ccp_dataset_to_tfrecord import *

class CustomModel(Model):
  def __init__(self,
               feature_dim,
               embed_dim,
               fc_dims,
               dropout,
               expert_num,
               task_num,
               expert_hidden_layers, 
               gate_hidden_layers,
               expert_activation,
               gate_activation
              ):
    super(CustomModel, self).__init__()
    self.embedding = Embedding(feature_dim + 1, embed_dim)
    self.din_for_category_seq = DeepInterestNet(feature_dim, embed_dim, fc_dims, dropout, self.embedding) #(seq_fea, target_ad_fea)
    self.din_for_shop_seq = DeepInterestNet(feature_dim, embed_dim, fc_dims, dropout, self.embedding) #(seq_fea, target_ad_fea)
    self.din_for_brand_seq = DeepInterestNet(feature_dim, embed_dim, fc_dims, dropout, self.embedding) #(seq_fea, target_ad_fea)
    self.din_for_intention_seq = DeepInterestNet(feature_dim, embed_dim, fc_dims, dropout, self.embedding) #(seq_fea, target_ad_fea)

    # ctr 、 cvr 塔各两层网络
    self.dense1 = Dense(64, activation='relu')
    self.dense2 = Dense(1, activation='sigmoid')

    self.dense3 =  Dense(64, activation='relu')
    self.dense4 =  Dense(1, activation='sigmoid')

    self.mmoe_layer = MyMMOE(expert_num, task_num, expert_hidden_layers, 
              gate_hidden_layers, expert_activation, gate_activation)

  def call(self, inputs):
    user_cate_seq = inputs['u_c_f']
    user_cate_seq_val = inputs['u_c_fv']
    user_cate_dft_fea = list(USER_CAT_SEQ_Fields.values())[0]

    user_shop_seq = inputs['u_s_f']
    user_shop_seq_val = inputs['u_s_fv']
    user_shop_dft_fea = list(USER_SHOP_SEQ_Fields.values())[0]

    user_brand_seq = inputs['u_b_f']
    user_brand_seq_val = inputs['u_b_fv']
    user_brand_dft_fea = list(USER_BRAND_SEQ_Fields.values())[0]

    user_intent_seq = inputs['u_i_f']
    user_intent_seq_val = inputs['u_i_fv']
    user_intent_dft_fea = list(USER_INTENTION_SEQ_Fields.values())[0]

    target_ad_fea = inputs['a_f']
    print("target_ad_fea: ", target_ad_fea)

    ad_cate_fea = target_ad_fea[:, 1]
    ad_shop_fea = target_ad_fea[:, 2]
    ad_brand_fea = target_ad_fea[:, 3]
    ad_intent_fea = inputs['a_i_f']

    target_ad_cate_fea = tf.expand_dims(ad_cate_fea, axis = -1)
    print("target_ad_cate_fea: ", target_ad_cate_fea)
    target_ad_cate_fea_emb = self.embedding(target_ad_cate_fea)# batch, 1, emb
    weighted_cate_seq_fea = self.din_for_category_seq(user_cate_seq, user_cate_seq_val, user_cate_dft_fea, target_ad_cate_fea_emb)

    target_ad_shop_fea = tf.expand_dims(ad_shop_fea, axis = -1)
    print("target_ad_shop_fea: ", target_ad_shop_fea)
    target_ad_shop_fea_emb = self.embedding(target_ad_shop_fea)# batch, 1, emb
    weighted_shop_seq_fea = self.din_for_shop_seq(user_shop_seq, user_shop_seq_val, user_shop_dft_fea, target_ad_shop_fea_emb)

    target_ad_brand_fea = tf.expand_dims(ad_brand_fea, axis = -1)
    print("target_ad_brand_fea: ", target_ad_brand_fea)
    target_ad_brand_fea_emb = self.embedding(target_ad_brand_fea)# batch, 1, emb
    weighted_brand_seq_fea = self.din_for_brand_seq(user_brand_seq, user_brand_seq_val, user_brand_dft_fea, target_ad_brand_fea_emb)

    # target_ad_intent_fea = tf.expand_dims(ad_intent_fea, axis = -1)
    print("ad_intent_fea: ", ad_intent_fea)

    # 由于target_ad_intent_fea 也是一个序列，这里在喂给din之前也进行pooling
    elements_equal_to_dft_val = tf.equal(ad_intent_fea, list(AD_SEQ_Fields.values())[0])
    elements_equal_to_dft_val_int = tf.cast(elements_equal_to_dft_val, tf.int32)
    target_ad_intent_fea_seq_len = tf.reduce_sum(elements_equal_to_dft_val_int, 1)

    longest_seq_len = tf.shape(ad_intent_fea)[1]
    boolean_mask = tf.sequence_mask(target_ad_intent_fea_seq_len, longest_seq_len) # batch, longest_seq_len
    mask = tf.cast(boolean_mask, tf.float32) # batch, longest_seq_len

    expanded_mask = tf.expand_dims(mask, axis = -1)
    print("expanded_mask:", expanded_mask)

    #target_ad_intent_fea是序列
    target_ad_intent_fea_emb = self.embedding(ad_intent_fea)
    print("target_ad_intent_fea_emb:", target_ad_intent_fea_emb)
    masked_target_ad_intent_fea_emb = target_ad_intent_fea_emb * expanded_mask
    pooled_target_ad_intent_fea = tf.reduce_sum(masked_target_ad_intent_fea_emb, axis=1)
    print("pooled_target_ad_intent_fea:", pooled_target_ad_intent_fea)

    pooled_target_ad_intent_fea = tf.expand_dims(pooled_target_ad_intent_fea, axis = 1)

    weighted_intent_seq_fea = self.din_for_intention_seq(user_intent_seq, user_intent_seq_val, user_intent_dft_fea, pooled_target_ad_intent_fea)

    embed_concat = tf.concat([weighted_cate_seq_fea, weighted_shop_seq_fea, weighted_brand_seq_fea, weighted_intent_seq_fea],axis=1)    # None * (F * K)
#, weighted_intent_seq_fea
    # embed_concat = weighted_cate_seq_fea
    print("embed_concat: ", embed_concat)

    mmoe_output = self.mmoe_layer(embed_concat)

    x = self.dense1(mmoe_output)
    print("x:", x)
    ctr_outputs = self.dense2(x)
    print("ctr_outputs:", ctr_outputs)

    x_1 = self.dense3(mmoe_output)
    cvr_outputs = self.dense4(x_1) #条件概率，表示在发生点击的条件下发生转化的概率

    ctcvr_outputs = ctr_outputs * cvr_outputs

    # ctr塔的label是点击
    # cvr塔没有label，cvr任务的预估值是两个塔的连乘结果，label是转化
    
    return [ctr_outputs, ctcvr_outputs]
