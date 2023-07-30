
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
# 和MMOE的区别
# expert分为共享expert和每个任务专有的expert
# 每个task有一个gate，该gate计算出来当前任务的所有expert的权重，然后对这些expert进行加权
# 加权输出的结果作为每个task的输出。


# 参考https://github.com/datawhalechina/fun-rec/blob/master/docs/ch02/ch2.2/ch2.2.5/PLE.md
# 这里的实现，CGC中还有一个共享的gate
# 最底层的CGC中每个gate和expert的输入是一样的
# 但是上层的CGC的输入都是前面的CGC的输出。
# 由于前面的CGC是每个task和共享task各有一个输入，所以下一层的CGC
# 中每个task中的expert和gate的输入都是前一层对应的task的输出。
# 所以，CGC的输入最好是分开的，比如有4个task，那么输入最好就是长度为5的list。

# 还有一点，共享task下是对所有task的expert进行加权，而其他task都是只对自己任务的expert进行加权。

# 这里还有一个实现：DeepCTR/deepctr/models/multitask/ple.py 
# 和上面的实现是一样的。。
class MyCGC(Layer):
  def __init__(self, shared_expert_num, 
               exclusive_expert_num_list, task_num, expert_hidden_layers, 
              gate_hidden_layers_list, expert_activation, gate_activation):
    super(MyCGC, self).__init__()
    self.shared_expert_num =  shared_expert_num
    self.exclusive_expert_num_list = exclusive_expert_num_list
    self.task_num = task_num
    self.expert_hidden_layers = expert_hidden_layers
    self.gate_hidden_layers_list = gate_hidden_layers_list
    self.expert_activation = expert_activation
    self.gate_activation = gate_activation

# 每个expert的参数是分开的，其实可以合在一起，合在一起之后可以更好利用GPU的并行计算能力，速度应该
# 会更快
  def build(self, input_shape):
    self.shared_expert_fc_layers = [tf.keras.Sequential()] * self.shared_expert_num

    # 可以增加Dropout层、激活函数层、BN层
    for i in range(self.shared_expert_num):
      input_dim = input_shape[-1] # batch_size, input_dim
      for fc_dim in self.expert_hidden_layers:
          self.shared_expert_fc_layers[i].add(Dense(fc_dim, input_shape = [input_dim,]))
          input_dim = fc_dim
    
    self.exclusive_expert_fc_layers = [] #双层expert，第一层是任务，第二层是任务下的expert
    for expert_num in self.exclusive_expert_num_list: #每个任务的expert的数量
      expert_fc_layers = [tf.keras.Sequential()] * expert_num
      for i in range(expert_num):
        input_dim = input_shape[-1] # batch_size, input_dim
        for fc_dim in self.expert_hidden_layers:
            expert_fc_layers[i].add(Dense(fc_dim, input_shape = [input_dim,]))
            input_dim = fc_dim
      self.exclusive_expert_fc_layers.append(expert_fc_layers)

    self.gate_fc_layers = [tf.keras.Sequential()] * self.task_num
    for i in range(self.task_num):
      # 每个任务的专家数量不一样，所以每个gate的输出维度可能不一样，为了灵活性，这里使用list保存每个task的gate隐层结构
      for gate_hidden_layers in self.gate_hidden_layers_list:
        input_dim = input_shape[-1]
        for fc_dim in gate_hidden_layers:
            self.gate_fc_layers[i].add(Dense(fc_dim, input_shape = [input_dim,]))
            input_dim = fc_dim

  def call(self, input):
    # 共享专家网络输出
    shared_expert_out = [] # expert_num, batch_size, output_dim
    for i in range(self.shared_expert_num):
      single_expert_out = self.shared_expert_fc_layers[i](input[-1])
      shared_expert_out.append(single_expert_out)
    # 专有专家网络输出
    exclusive_expert_out = []
    for i in range(self.task_num):
    # for expert_fc_layers in self.exclusive_expert_fc_layers: #每个任务的expert的网络
      expert_fc_layers = self.exclusive_expert_fc_layers[i]
      expert_output = []
      for expert_fc_layer in expert_fc_layers: # 每个 expert
        expert_output.append(expert_fc_layer(input[i]))
      exclusive_expert_out.append(expert_output)
    
    # 将每个任务的专有专家网络输出和共享专家网络输出concat在一起
    batch_dim_concated_expert_output = []
    for expert_output in exclusive_expert_out:
      concated_expert_output = tf.concat([shared_expert_out, expert_output], axis = 0) #expert_num, batch, hidden_size
      batch_dim_concated_expert_out_tensor = tf.transpose(concated_expert_output, perm = [1, 0, 2])
      batch_dim_concated_expert_output.append(batch_dim_concated_expert_out_tensor)

    # expert_out_tensor = tf.concat(expert_out, axis = 0)
    gate_out = []
    for i in range(self.task_num):
      single_gate_out = self.gate_fc_layers[i](input[i])
      gate_out.append(single_gate_out)

    weighted_output = []
    for i in range(self.task_num):
      gate_weight = gate_out[i]
      gate_weight = tf.expand_dims(gate_weight, -1) #batch_size, expert_num, 1

      expert_output = batch_dim_concated_expert_output[i]

      task_i_output = gate_weight * expert_output

      task_i_output = tf.reduce_sum(task_i_output, axis = 1) # batch_size, output_dim
      weighted_output.append(task_i_output)

    # 还缺一个共享task
    return weighted_output
