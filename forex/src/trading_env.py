#this defines the trading MDP
from data_preprocessing import Data
from agent import Agent
import numpy as np
T = 96

# state here means a sequence of Data's states of length T
# steps are done T days ahead
class TradingEnv:
    def __init__(self, data: Data, initial_value=100000):
        self.initial_value = initial_value
        self.portfolio = [initial_value]
        self.actions = []
        self.history = []
        self.prev_close = None
        self.data = data
        self.spread = 0.08
        self.trade_size = 100000

    # Returns: state
    def reset(self) -> object:
        self.portfolio = [self.initial_value]
        self.history = []
        self.data.reset()
        closing, state_initial = self.data.next()
        self.prev_close = closing
        return state_initial

    # Returns: actions, rewards, new_states, selected new_state, done
    def step(self, action) -> object:
        actions = [-1, 0, 1]
        v_old = self.portfolio[-1]

        try:
            closing, state_next = self.data.next()
            done = False
        except:
            state_next = None
            done = True

        new_states = []
        for a in actions:
            a_ = np.zeros(3, dtype=np.float32)
            a_[a+1] = 1.
            new_states.append(np.append(state_next, a_))

        current_closed = closing
        if self.prev_close is not None:
            current_open = self.prev_close
            self.prev_close = current_closed
        else:
            raise Exception("No previous close price saved!")

        v_new = []
        for a in actions:
            commission = self.trade_size * np.abs(a - self.actions[-1]) * self.spread
            v_new[a+1] = v_old + a * self.trade_size * (current_closed - current_open) - commission

        v_new = np.asarray(v_new)
        rewards = np.log(v_new/v_old)

        self.actions.append(action+1)
        self.portfolio.append(v_new[action+1])

        return actions, rewards, new_states, new_states[action+1], done

    def print_stats(self):
        pass


class RunAgent:
    def __init__(self, env: TradingEnv, agent: Agent):
        self.env = env
        self.agent = agent

    def run(self, episodes):
        self.agent.initialize()

        state = self.env.reset() # initial_state

        for step in range(episodes):
            action = self.agent.get_action(state) # select greedy action, exploration is done in step-method

            actions, rewards, new_states, state, done = self.env.step(action)

            if done:
                self.env.print_stats()
                break

            self.agent.store(state, actions, rewards, new_states)

            if self.agent.is_memory_filled() and step % self.agent.network.T == 0: # TODO: Check if this is the right T
                self.agent.train(update=True)

        self.env.print_stats()