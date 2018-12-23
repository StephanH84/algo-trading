from data_preprocessing import Data
from trading_env import TradingEnv, RunAgent
from agent import Agent
from network import Network

DIRECTORY = "F:/Dev/Data/truefx/"

def main():
    T = 96
    M = 1 # minibatch size
    alpha = 0.00025 # Learning rate
    gamma = 0.99 # Discount factor
    theta = 0.001 # Target network

    RunAgent(TradingEnv(Data(DIRECTORY)), Agent(Network(T, M, alpha, gamma, theta))).run(100)

    # weight initialization!!

main()