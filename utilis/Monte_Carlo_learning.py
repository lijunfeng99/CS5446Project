import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

from utilis.agentbase import Agent

class MonteCarloAgent(Agent):
    def __init__(self, model_name="MonteCarlo"):
        super().__init__(model_name)
        
        self.returns = {}  # Cache returns for state-action pairs
        self.returns_count = {}  # Count of visits to state-action pairs
        self.policy = {}  # Policy to be improved

    def __call__(self, observation, configuration):
        state = observation.board
        state = tuple(state)
        if state not in self.policy.keys():
            self.policy[state] = random.choice([c for c in range(configuration.columns) if observation.board[c] == 0])
        return self.policy[state]

    def update_policy(self, episode):
        # Calculate returns and update policy based on the returns
        G = 0
        for observation, action, reward in reversed(episode):
            G = reward + self.config.GAMMA * G
            state = observation.board
            state = tuple(state)
            for i in range(self.config.num_actions):
                if (state, i) not in self.returns:
                    self.returns[(state, i)] = 0
                    self.returns_count[(state, i)] = 0
            
            self.returns[(state, action)] += G
            self.returns_count[(state, action)] += 1
            # Update policy based on average returns
            q_values = [self.returns[(state, i)] / self.returns_count[(state, i)] if self.returns_count[(state, i)] > 0 else self.config.INVALID_MOVE_PENALTY for i in range(self.config.num_actions)]
            self.policy[state] = np.argmax(q_values)

    def train(self):
        train_env = self.env.train([None, "random"]) # random, negamax
        self.epsilon = self.config.EPSILON_START
        win = 0
        sum_reward = 0

        tbar = tqdm(range(self.config.NUM_EPISODES))
        for episode in tbar:
            state = train_env.reset()
            done = False
            total_reward = 0
            one_episode = []

            while not done:
                # Random action
                action = random.choice([c for c in range(self.config.num_actions) if state.board[c] == 0])

                next_state, reward, done, _ = train_env.step(action)
                if reward == None:
                    reward = self.config.INVALID_MOVE_PENALTY # Invalid move penalty
                one_episode.append((state, action, reward))

                state = next_state
                total_reward += reward

            win += 1 if reward > 0 else 0
            self.rewards.append(total_reward)
            sum_reward += total_reward

            # Update policy
            self.update_policy(one_episode)

            tbar.set_postfix(Reward= round(sum_reward / (episode + 1), 3), win= round(win / (episode + 1), 3))