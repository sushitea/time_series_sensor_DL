import tensorflow as tf

class conv1D(tf.keras.Model):

  def __init__(self):
    super(conv1D, self).__init__()
    self.conv1 = tf.keras.layers.Conv1D(64,3,activation=tf.nn.relu,input_shape=(24,13))
    self.conv2 = tf.keras.layers.Conv1D(128,3,activation=tf.nn.relu)
    self.conv3 = tf.keras.layers.Conv1D(256,3,activation=tf.nn.relu)
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(1024,activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(18,activation=tf.nn.softmax)
    
  def call(self, inputs, training=False):
    x = self.conv1(inputs)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    outputs = self.dense2(x)

    return outputs

def test_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(24,13)))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(18, activation='softmax'))
    return model

    