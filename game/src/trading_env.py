#this defines the trading MDP
from data_preprocessing import Data
from agent import Agent
T = 96

# state here means a sequence of Data's states of length T
# steps are done T days ahead
class TradingEnv:
    def __init__(self, data: Data):
        pass

    # Returns: state
    def reset(self) -> object:
        return None

    # Returns: actions, rewards, new_states, state, done
    def step(self, action) -> object:
        return None

    def print_stats(self):
        pass


class RunAgent:
    def __init__(self, env: TradingEnv, agent: Agent):
        self.env = env
        self.agent = agent

    def run(self, episodes):
        self.agent.initialize()

        state = self.env.reset() #initial_state

        for step in range(episodes):
            action = self.agent.get_action(state) #select greedy action

            actions, rewards, new_states, state, done = self.env.step(action)

            if done:
                self.env.print_stats()
                break

            self.agent.store(state, actions, rewards, new_states)

            if self.agent.is_memory_filled() and step % T == 0:
                self.agent.train(update=True)

        self.env.print_stats()