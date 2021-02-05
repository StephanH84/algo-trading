from data_preprocessing import Data
from trading_env import TradingEnv, RunAgent
from agent import Agent
from network import Network

DIRECTORY = "F:/Dev/Data/truefx/"

def main():
    T = 96
    M = 16 # minibatch size
    alpha = 0.00025 # Learning rate
    gamma = 0.99 # Discount factor
    theta = 0.001 # Target network
    n_units = 32 # number of units in a hidden layer

    RunAgent(TradingEnv(Data(DIRECTORY, T)), Agent(Network(T, M, alpha, gamma, theta, n_units))).run(5*T)

    # weight initialization!!

main()