# 参考https://github.com/Prayforhanluo/CTR_Algorithm/blob/main/tensorflow2.0/model/DIN.py

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, Dropout
class DeepInterestNet(Layer):
    """
        Deep Interest Net
    """
    def __init__(self, feature_dim, embed_dim, fc_dims, dropout, embedding_table):
        super(DeepInterestNet, self).__init__()
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
        # 输入维度为(batch_size, input_length) -> (batch_size, input_length, output_dim)
        self.fc_layers = tf.keras.Sequential()
        input_dim = embed_dim * 4
        for fc_dim in fc_dims:
            self.fc_layers.add(Dense(fc_dim, input_shape = [input_dim,]))
            # self.fc_layers.add(Dice())
            self.fc_layers.add(Dropout(dropout))
            input_dim = fc_dim
        self.fc_layers.add(Dense(1, input_shape = [input_dim,]))

        self.embedding = embedding_table
        self.embed_dim = embed_dim
    
    def call(self, seq_fea, seq_fea_val, seq_dft_fea, target_ad_fea_emb):
        """
            seq_fea : batch * (seq_len)
            target_ad_fea : batch * ad_fea_len
            x : (behaviors * 40, ads * 1) -> batch * (behaviors + ads)
        """
        # define mask
        # behavior_x = x[:,:-1]
        print("seq_fea: ", seq_fea)
        # seq_len = tf.math.count_nonzero(seq_fea, 1) # N
        # seq_len = tf.cast(seq_len, tf.int32)
        elements_equal_to_dft_val = tf.equal(seq_fea, seq_dft_fea)
        elements_equal_to_dft_val_int = tf.cast(elements_equal_to_dft_val, tf.int32)
        seq_len = tf.reduce_sum(elements_equal_to_dft_val_int, 1)
        print("seq_len:", seq_len)
        # longest_seq_len = tf.math.reduce_max(seq_len)
        # seq_len = tf.shape(seq_fea)[1]
        # print("seq_len: ", seq_len)
        # longest_seq_len = tf.math.reduce_max(tf.shape(seq_fea))
        # longest_seq_len = tf.cast(longest_seq_len, tf.int32)

        longest_seq_len = tf.shape(seq_fea)[1]
        # longest_seq_len_2 = tf.expand_dims(longest_seq_len, 0)

        print("longest_seq_len: ", longest_seq_len)
        # print("longest_seq_len_2: ", longest_seq_len_2)
        # max_len = tf.reduce_max(tf.shape(inputs))


        # batch, longest_seq_len
        # keys = tf.keras.preprocessing.sequence.pad_sequences(seq_fea, padding='post', maxlen=longest_seq_len)
        # https://stackoverflow.com/questions/42334646/tensorflow-pad-unknown-size-tensor-to-a-specific-size
        # 
        print("seq_fea: ", seq_fea)
        print(longest_seq_len)
        # paddings = tf.constant([[0, 0], [0, longest_seq_len - seq_len]])
        #     ValueError: Tried to convert 'paddings' to a tensor and failed. Error: Shapes must be equal rank, but are 0 and 1
        # 下面这段代码报下面这个错误
        # From merging shape 0 with other shapes. for '{{node deep_interest_net/Pad/packed/1}} = Pack[N=2, T=DT_INT32, axis=0](deep_interest_net/Pad/packed/1/0, deep_interest_net/sub)' with input shapes: [], [1].
        # padded_inputs = tf.pad(seq_fea, [[0, 0], [0, longest_seq_len_2 - tf.shape(seq_fea)[1]]],'CONSTANT', constant_values = 0)
        # padded_inputs = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [0, longest_seq_len - tf.shape(x)[1]]],'CONSTANT', constant_values = 0 ))(seq_fea)
        # print("padded_inputs:", padded_inputs)

        keys_emb = self.embedding(seq_fea) # batch, longest_seq_len, emb
        print("keys_emb: ", keys_emb)

        seq_fea_val_expanded = tf.expand_dims(seq_fea_val, axis = -1)
        weighted_keys_emb = keys_emb * seq_fea_val_expanded

        # return keys_emb

        # mask = tf.cast(behavior_x > 0, tf.float32)
        # seq_len_expanded = tf.expand_dims(seq_len, axis=-1)
        # https://www.tensorflow.org/api_docs/python/tf/sequence_mask
        # batch, longest_seq_len
        boolean_mask = tf.sequence_mask(seq_len, longest_seq_len) # batch, longest_seq_len
        mask = tf.cast(boolean_mask, tf.int32) # batch, longest_seq_len
        print("mask:", mask)
        # if len(target_ad_fea.shape) == 2:
            # target_ad_fea = tf.expand_dims(target_ad_fea[:, 0], axis = -1)
            # print("target_ad_fea: ", target_ad_fea)
        # else:
        #     tf.nn.embedding_lookup_sparse(target_ad_fea) 
        print("target_ad_fea_emb: ", target_ad_fea_emb)
        # queries = tf.concat([target_ad_fea_emb]*longest_seq_len_new, axis = 1) #batch, longest_seq_len, emb
        queries = tf.broadcast_to(target_ad_fea_emb, [tf.shape(target_ad_fea_emb)[0], longest_seq_len, self.embed_dim])
        print("queries %s"%(queries.get_shape()))

        attn_input = tf.concat([queries,
                            weighted_keys_emb,
                            queries - weighted_keys_emb,
                            queries * weighted_keys_emb], axis = -1)
        att_score = self.fc_layers(attn_input) # batch, seq_len, 1
        print("att_score: %s"%(att_score.get_shape()))

        padding_score = tf.ones_like(att_score) * (-2 ** 32 + 1)

        outputs = tf.where(tf.expand_dims(boolean_mask, axis=-1), att_score, padding_score)

        normalized_outputs = tf.nn.softmax(outputs) #batch, longest_seq_len, 1
        normalized_outputs = tf.reshape(normalized_outputs,[-1, 1, tf.shape(normalized_outputs)[1]]) #batch, 1, longest_seq_len

        weighted_outputs = tf.matmul(normalized_outputs, weighted_keys_emb) # (batch, 1, longest_seq_len)* (#batch, longest_seq_len, 8) -> (batch,1, 8)
        print('weighted_output', weighted_outputs)
        weighted_outputs = tf.reduce_sum(weighted_outputs, 1)
        print('weighted_output', weighted_outputs)
        return weighted_outputs