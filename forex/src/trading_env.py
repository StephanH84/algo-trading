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

    def merge_state_action(self, state, a_variable):
        T = len(state)
        actions_for_state = self.actions[self.data.n: self.data.n+T-2]
        actions_for_state.append(a_variable)
        actions_encoded = map(lambda a: self.hot_encoding(a), actions_for_state)

        result = []
        for s, a in zip(state, actions_encoded):
            new_s = s[:]
            new_s.extend(a)
            result.append(new_s)

        result = np.asarray(result)
        return result

    # Returns: state
    def reset(self) -> object:
        self.portfolio = [self.initial_value]
        self.history = []
        self.data.reset()
        closing, state_initial = self.data.next()
        self.prev_close = closing
        return np.append(np.asarray(state_initial), self.hot_encoding(0))

    # Returns: actions, rewards, new_states, selected new_state, done
    def step(self, action) -> object: # TODO: check if everything is correct esp. with the action values and array indexing
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
            new_states.append(self.merge_state_action(state_next, a))

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

        self.actions.append(action)
        self.portfolio.append(v_new[action+1])

        return actions, rewards, new_states, new_states[action+1], done

    def hot_encoding(self, a):
        a_ = np.zeros(3, dtype=np.float32)
        a_[a + 1] = 1.
        return a_

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