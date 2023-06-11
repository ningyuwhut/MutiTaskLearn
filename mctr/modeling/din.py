# 参考https://github.com/Prayforhanluo/CTR_Algorithm/blob/main/tensorflow2.0/model/DIN.py

import tf.keras.Layers as Layers
import tf.keras.Model as Model
class DeepInterestNet(Model):
    """
        Deep Interest Net
    """
    def __init__(self, feature_dim, embed_dim, mlp_dims, dropout):
        super(DeepInterestNet, self).__init__()
        self.embedding = Layers.Embedding(feature_dim+1, embed_dim)
        self.AttentionActivate = AttentionPoolingLayer(embed_dim, dropout)
          
        self.fc_layers = keras.Sequential()
        input_dim = embed_dim*2
        for fc_dim in mlp_dims:
            self.fc_layers.add(Layers.Dense(fc_dim, input_shape = [input_dim,]))
            self.fc_layers.add(Layers.Activation('relu'))
            self.fc_layers.add(Layers.Dropout(dropout))
            self.input_dim = fc_dim
        self.fc_layers.add(Layers.Dense(1, input_shape = [input_dim,]))  
    
    def call(self, inputs):
        """
            x : (behaviors * 40, ads * 1) -> batch * (behaviors + ads)
        """
        # define mask
        behavior_x = x[:,:-1]
        mask = tf.cast(behavior_x > 0, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        
        ads_x = x[:,-1]
        
        # embedding
        query_ad = tf.expand_dims(self.embedding(ads_x), axis=1)
        user_behavior = self.embedding(behavior_x)
        user_behavior = user_behavior * mask
        
        # attn pooling
        user_interest = self.AttentionActivate(query_ad, user_behavior, mask)
        concat_input = tf.concat([user_interest, 
                                  tf.squeeze(query_ad, axis=1)],
                                 axis = 1)
        # MLPs prediction
        out = self.fc_layers(concat_input)
        out = tf.sigmoid(out)
        
        return out