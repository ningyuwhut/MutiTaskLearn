
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout
class MyMMOE(Layer):
  def __init__(self, expert_num, task_num, expert_hidden_layers, 
              gate_hidden_layers, expert_activation, gate_activation):
    super(MyMMOE, self).__init__()
    self.expert_num =  expert_num
    self.task_num = task_num
    self.expert_hidden_layers = expert_hidden_layers
    self.gate_hidden_layers = gate_hidden_layers
    self.expert_activation = expert_activation
    self.gate_activation = gate_activation


# 每个expert的参数是分开的，其实可以合在一起，合在一起之后可以更好利用GPU的并行计算能力，速度应该
# 会更快
  def build(self, input_shape):
    self.expert_fc_layers = [tf.keras.Sequential()] * self.expert_num

    # 可以增加Dropout层、激活函数层、BN层
    for i in range(self.expert_num):
      input_dim = input_shape[-1] # batch_size, input_dim
      for fc_dim in self.expert_hidden_layers:
          self.expert_fc_layers[i].add(Dense(fc_dim, input_shape = [input_dim,]))
          input_dim = fc_dim
      # self.expert_fc_layers[i].add(Dense(1, input_shape = [input_dim,]))

    self.gate_fc_layers = [tf.keras.Sequential()] * self.task_num
    for i in range(self.task_num):
      input_dim = input_shape[-1]
      for fc_dim in self.gate_hidden_layers:
          self.gate_fc_layers[i].add(Dense(fc_dim, input_shape = [input_dim,]))
          input_dim = fc_dim
      # self.gate_fc_layers[i].add(Dense(1, input_shape = [input_dim,]))

  def call(self, input):
    expert_out = [] # expert_num, batch_size, output_dim
    for i in range(self.expert_num):
      single_expert_out = self.expert_fc_layers[i](input)
      expert_out.append(single_expert_out)
    expert_out_tensor = tf.concat(expert_out, axis = 0)
    batch_dim_expert_out_tensor = tf.transpose(expert_out_tensor, perm = [1, 0, 2])
    gate_out = []
    for i in range(self.task_num):
      single_gate_out = self.gate_fc_layers[i](input)
      gate_out.append(single_gate_out)

    weighted_output = []
    for i in range(self.task_num):
      gate_weight = gate_out[i]
      gate_weight = tf.expand_dims(gate_weight, -1) #batch_size, expert_num, 1

      task_i_output = gate_weight * batch_dim_expert_out_tensor # batch_size, expert_num, output_dim

      task_i_output = tf.reduce_sum(task_i_output, axis = 1) # batch_size, output_dim
      weighted_output.append(task_i_output)
    return weighted_output
