
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, Dropout
class MyMMOE(Layer):
  def __init__(self, expert_num, task_num, expert_hidden_layers, 
              gate_hidden_layers):
    super(MyMMOE, self).__init__()
    self.expert_num =  expert_num
    self.task_num = task_num
    self.expert_hidden_layers = expert_hidden_layers
    self.gate_hidden_layers = gate_hidden_layers

    self.fc_layers = tf.keras.Sequential()
    input_dim = embed_dim * 4
    for fc_dim in fc_dims:
        self.fc_layers.add(Dense(fc_dim, input_shape = [input_dim,]))
        # self.fc_layers.add(Dice())
        self.fc_layers.add(Dropout(dropout))
        input_dim = fc_dim
    self.fc_layers.add(Dense(1, input_shape = [input_dim,]))

  def call(self, input):
    expert_out = []
    for i in range(self.expert_num):
      for hidden_dim in self.expert_hidden_layers:
