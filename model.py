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
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(24,113)))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(18, activation='softmax'))
    return model

class convLSTM(tf.keras.Model):

  def __init__(self):
    super(convLSTM, self).__init__()
    self.conv1 = tf.keras.layers.Conv1D(64,5,activation=tf.nn.relu,input_shape=(24,113))
    self.conv2 = tf.keras.layers.Conv1D(64,5,activation=tf.nn.relu)
    self.conv3 = tf.keras.layers.Conv1D(64,5,activation=tf.nn.relu)
    self.conv4 = tf.keras.layers.Conv1D(64,5,activation=tf.nn.relu)
    self.lstm1 = tf.keras.layers.LSTM(128,return_sequences=True)
    self.lstm2 = tf.keras.layers.LSTM(128)
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(18,activation=tf.nn.softmax)
    
  def call(self, inputs, training=False):
    x = self.conv1(inputs)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    
    x = self.lstm1(x)
    x = self.lstm2(x)
    x = self.flatten(x)
    x = self.dense1(x)
    
    return x

def test_model_1():
  model = Sequential()
  model.add(TimeDistributed(Conv1D(filters=8, kernel_size=2, activation='relu', padding='same'), batch_input_shape=(None, None, 4, 18)))
  model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
  model.add(TimeDistributed(Flatten()))
  model.add(LSTM(16))
  model.add(Dense(9, activation='softmax'))

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None, 24, 113)))
  model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)))
  model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
  model.add(tf.keras.layers.LSTM(50, activation='relu'))
  model.add(tf.keras.layers.Dense(1))
  return model
    