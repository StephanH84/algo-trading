import numpy as np
import tensorflow as tf
from common import hot_encoding

import unittest

class Network:
    def __init__(self, T: int, M: int, alpha: float, gamma: float, theta: float):
        data_size = 8+3+3
        self.gamma = gamma
        self.M = M
        self.data_size = data_size
        self.T = T
        self.alpha = alpha
        self.theta = theta

        self.initialize()

    def initialize(self):
        data_size, M, T, alpha = self.data_size, self.M, self.T, self.alpha
        state = tf.placeholder(tf.float32, [None, T, data_size])
        y = tf.placeholder(tf.float32, [None, 1])
        action = tf.placeholder(tf.float32, [None, 3])

        def q_network(input):
            input_reshaped = tf.reshape(input, [-1, data_size])
            h1 = tf.layers.dense(input_reshaped, 256, activation="elu",
                                 kernel_initializer=tf.initializers.he_normal(),
                                 bias_initializer=tf.initializers.zeros)
            h2 = tf.layers.dense(h1, 256, activation="elu",
                                 kernel_initializer=tf.initializers.he_normal(),
                                 bias_initializer=tf.initializers.zeros)
            h2_reshaped = tf.reshape(h2, [-1, T, 256])
            h2_reshuffled = tf.transpose(h2_reshaped, perm=[1, 0, 2])
            # need to shuffle to shape TxMx256 to

            cell = tf.nn.rnn_cell.LSTMCell(256, name='basic_lstm_cell', initializer=tf.initializers.identity)
            fused_cell = tf.contrib.rnn.FusedRNNCellAdaptor(cell, True)
            cell_output, states = fused_cell(h2_reshuffled, dtype=tf.float32)

            return tf.layers.dense(cell_output[-1], 3, activation="softmax",
                                bias_initializer=tf.initializers.zeros,
                                kernel_initializer=tf.initializers.random_normal(0, 0.001))

        q_value = q_network(state)  # tensor of shape [None, 3]

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
        # q_value_next = q_network(state_next)
        state, action, reward, state_next = np.swapaxes(np.asarray(batch), 1, 0)
        state = np.asarray(state.tolist())
        state_next = np.asarray(state_next.tolist())

        # CALCULATE y:
        # calculate the q_value for state_next with online weights
        action_max = self.sess.run(self.output_action, feed_dict={self.state: state_next})

        feed_dict = {self.state: state_next}
        for var, val in zip(self.tvars, self.target_weights):
            feed_dict[var] = val

        q_value_next = self.sess.run(self.q_value, feed_dict=feed_dict)
        # y = reward + gamma * q_value_next
        y = reward + self.gamma * np.asarray([q_value_next[i, action_max[i]] for i in range(self.M)])

        action = np.asarray([hot_encoding(n) for n in action])
        y = np.expand_dims(y, axis=1)
        feed_dict = {self.state: state, self.action: action, self.y: y}
        _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)

    def evaluate(self, state):
        return self.sess.run(self.output_action, feed_dict={self.state: np.expand_dims(state, axis=0)})[0]

    def update(self):
        # so-called soft update
        new_vars = self.sess.run(self.tvars)
        new_target_weights = []
        for old_var, new_var in zip(self.target_weights, new_vars):
            new_target_weights.append((1 - self.theta) * old_var + self.theta * new_var)
        self.target_weights = new_target_weights