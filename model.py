import tensorflow as tf

class conv1D(tf.keras.Model):

  def __init__(self):
    super(conv1D, self).__init__()
    self.conv1 = tf.keras.layers.Conv1D(64,3,activation=tf.nn.relu,input_shape=(13,1))
    self.conv2 = tf.keras.layers.Conv1D(128,3,activation=tf.nn.relu)
    self.conv3 = tf.keras.layers.Conv1D(256,3,activation=tf.nn.relu)
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    

  def call(self, inputs, training=False):
    x = self.conv1(inputs)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    outputs = self.dense2(x)

    return outputs
