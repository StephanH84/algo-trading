from network import Network
import numpy as np
import random

class Agent:
    def __init__(self, network: Network):
        self.network = network
        self.memory = [] #replay memory
        self.N = 480

    def initialize(self):
        self.network.initialize()
        self.memory = []

    def get_action(self, state):
        return self.network.evaluate(state) - 1

    def store(self, state, actions, rewards, new_states):
        self.memory.extend([(state, actions[n], rewards[n], new_states[n]) for n in range(len(actions))])
        # truncate
        self.memory = self.memory[-self.N:]

    def is_memory_filled(self) -> bool:
        return len(self.memory) > self.N / 2

    def train(self, update=False):
        # sample random minibatch of state, action, rewards, next_states tuples
        n = len(self.memory)
        minibatch = [self.memory[i] for i in np.random.permutation(n)[0:self.network.M]]

        self.network.train(minibatch)
        if update:
            self.network.update()