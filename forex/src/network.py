import numpy as np
import tensorflow as tf

import unittest

class Network:
    def __init__(self, T: int, M: int, alpha: float, gamma: float, theta: float):
        data_size = 198
        self.gamma = gamma
        self.M = M
        self.data_size = data_size
        self.T = T
        self.alpha = alpha
        self.theta = theta

        self.initialize()

    def initialize(self):
        data_size, M, T, alpha = self.data_size, self.M, self.T, self.alpha
        state = tf.placeholder(tf.float32, [M, T, data_size])
        y = tf.placeholder(tf.float32, [M, 1])
        action = tf.placeholder(tf.float32, [M, 3])

        def q_network(input):
            input_reshaped = tf.reshape(input, [-1, data_size])
            h1 = tf.layers.dense(input_reshaped, 256, activation="elu")
            h2 = tf.layers.dense(h1, 256, activation="elu")
            h2_reshaped = tf.reshape(h2, [M, T, 256])

            cell = tf.contrib.rnn.BasicLSTMCell(256)
            state = cell.zero_state(M, dtype=tf.float32)
            for t in range(T):
                cell_output, state = cell(h2_reshaped[:, t, :], state)

            return tf.layers.dense(cell_output, 3, activation="softmax")

        q_value = q_network(state)  # tensor of shape [M, 3]

        q_value_a = tf.reduce_sum(q_value * action, axis=[1])
        loss = tf.reduce_sum(tf.squared_difference(y, q_value_a))

        train_step = tf.train.AdamOptimizer(alpha).minimize(loss)
        self.output_action = tf.argmax(q_value, axis=1)

        self.state = state
        self.y = y
        self.action = action
        self.train_step = train_step
        self.loss = loss
        self.q_value = q_value

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # save target weights:
        self.tvars = tf.trainable_variables()
        self.target_weights = self.sess.run(self.tvars)

    def train(self, batch):
        # state_next = tf.placeholder(tf.float32, [M, T, data_size])
        # reward = tf.placeholder(tf.float32, [M, 1])

        #transform batch of tuples to tuple of batches
        state, action, reward, state_next = np.swapaxes(np.asarray(batch), 1, 0)

        # CALCULATE y:
        # calculate the q_value for state_next with online weights
        action_max = self.sess.run(self.output_action, feed_dict={self.state: state_next})[0]

        feed_dict = {self.state: state_next}
        for var, val in zip(self.tvars, self.target_weights):
            feed_dict[var] = val

        # q_value_next = q_network(state_next)
        q_value_next = self.sess.run(self.q_value, feed_dict=feed_dict)[0]
        # y = reward + gamma * q_value_next
        y = reward + self.gamma * np.asarray([q_value_next[i, action_max[i]] for i in range(self.M)])

        feed_dict = {self.state: state, self.action: action, self.y: y}
        _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)

    def evaluate(self, state):
        return self.sess.run(self.output_action, feed_dict={self.state: state})[0]

    def update(self):
        # so-called soft update
        self.target_weights = (1 - self.theta) * self.target_weights + self.theta * self.sess.run(self.tvars)


class TestNetwork(unittest.TestCase):
    def test_init(self):
        net = Network(96, 64, 0.00025, 0.99, 0.001)
        net.initialize()


#from keras.models import Sequential
#from keras.layers import LSTM, Dense, Activation, Reshape
#from keras.optimizers import Adam
#from keras.losses import categorical_crossentropy

# model = Sequential()
# model.add(Dense(256, input_dim=(data_size,), batch_input_shape=(M * T, data_size)))
# model.add(Activation("elu"))
# model.add(Dense(256))
# model.add(Activation("elu"))
# model.add(Reshape((T, 256)))
# model.add(LSTM(256, input_shape=(T, 256)))
# model.add(Dense(3))
# model.add(Activation("softmax"))
# self.model = model
#
# model.compile(Adam, loss=categorical_crossentropy)
