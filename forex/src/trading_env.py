#this defines the trading MDP
from data_preprocessing import Data
from agent import Agent
import numpy as np
from copy import deepcopy
from common import hot_encoding, save_data_structure, DIRECTORY

# state here means a sequence of Data's states of length T
# steps are done T days ahead
class TradingEnv:
    def __init__(self, data: Data, initial_value=100000):
        self.initial_value = initial_value
        self.portfolio = [float(initial_value)]
        self.actions = []
        self.prev_close = None
        self.data = data
        self.spread = 0.08
        self.trade_size = 100000

    def merge_state_action(self, state, a_variable):
        T = len(state)
        actions_for_state = self.actions[self.data.n:][:T-1] # TODO: Check indices
        actions_for_state.append(a_variable)
        # TODO: Is there a better way
        diff = T - len(actions_for_state)
        if diff > 0:
            actions_for_state.extend([a_variable] * diff)

        result = []
        for s, a in zip(state, actions_for_state):
            new_s = deepcopy(s)
            new_s.extend(hot_encoding(a))
            result.append(new_s)

        result = np.asarray(result)
        return result

    # Returns: state
    def reset(self) -> object:
        self.portfolio = [float(self.initial_value)]
        self.data.reset()
        self.actions.append(0) # TODO: Check if this correct
        closing, state_initial = self.data.next()
        self.prev_close = closing
        return self.merge_state_action(state_initial, 0)

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
            v_new.append(v_old + a * self.trade_size * (current_closed - current_open) - commission)

        v_new = np.asarray(v_new)
        rewards = np.log(v_new/v_old)

        self.actions.append(int(action))
        self.portfolio.append(float(v_new[action+1]))

        return actions, rewards, new_states, new_states[action+1], done

    def print_stats(self):
        # draw portfolio and actions against price curve
        s = np.random.randint(0, 1000)
        save_data_structure(self.actions, DIRECTORY + "actions%s.json" % s )
        save_data_structure(self.portfolio, DIRECTORY + "portfolio%s.json" % s)


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
                break

            self.agent.store(state, actions, rewards, new_states)

            if self.agent.is_memory_filled() and step % self.agent.network.T == 0: # TODO: Check if this is the right T
                self.agent.train(update=True)

        self.env.print_stats()