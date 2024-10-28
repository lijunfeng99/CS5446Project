import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from kaggle_environments.utils import Struct
from kaggle_environments import make
from utilis.config import Config

MODEL_PATH = "./model/"

class Agent(nn.Module):
    def __init__(self, model_name : str):
        super().__init__()
        self.MODEL_NAME = model_name
        self.model_path = os.path.join(MODEL_PATH, self.MODEL_NAME)
        self.config = Config(MODEL_NAME= model_name, auto_load=True)
        self.env = make(self.config.env_name, debug=False)

        # Training records
        self.loss = []
        self.rewards = []
    
    def __call__(self, observation, configuration):
        '''
        Input:
        observation is a dictionary containing the current state of the game, including the board, remaining overage time, and the current player
        observation.board is a list of integers representing the current state of the board, with 0 indicating an empty cell, 1 indicating player 1's token, and 2 indicating player 2's token.
        configuration is a dictionary containing the configuration of the game, including the number of rows, columns, and the number of in-a-row needed to win.
        Output:
        action is an integer representing the column to place the token in.
        '''
        return 0
    
    def forward(self, x):
        return x

    def save(self):
        '''
        Save the agent's model to a file.
        '''
        self.to('cpu')
        os.makedirs(self.model_path, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(self.model_path, "model.pkl"))

        # 遍历每一个变量
        for key in self.__dict__:
            if isinstance(self.__dict__[key], list):
                with open(os.path.join(self.model_path, str(key) + ".npy"), 'wb') as f:
                    np.save(f, np.array(self.__dict__[key]))
    
    def load(self):
        '''
        Load the agent
        '''
        file_list = os.listdir(self.model_path)
        self.load_state_dict(torch.load(os.path.join(self.model_path, "model.pkl")))
        for file in file_list:
            if file.endswith(".npy"):
                key = file.split(".")[0]
                with open(os.path.join(self.model_path, file), 'rb') as f:
                    self.__dict__[key] = np.load(f)

        self.to(self.config.DEVICE)
    
    def test(self):
        '''
        Test the agent's performance on the environment.
        '''
        observation = {'remainingOverageTime': 60, 'step': 24, 'board': [0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 2, 1, 2, 0, 2, 0, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1], 'mark': 1}
        configuration = {'episodeSteps': 1000, 'actTimeout': 2, 'runTimeout': 1200, 'columns': 7, 'rows': 6, 'inarow': 4, 'agentTimeout': 60, 'timeout': 2}
        action = self(Struct(**observation), Struct(**configuration))
        return action

    def train(self):
        '''
        Train the agent's model.
        '''
        pass

    def draw_training_records(self):
        '''
        Draw the training records.
        Two plots are drawn: loss and rewards.
        '''
        fig = plt.figure(figsize=(10, 5))

        loss_avg = [sum(self.loss[i:i+100])/100 for i in range(len(self.loss)-100)]
        X = range(len(self.loss))
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(X, self.loss)
        plt.plot(X[100:], loss_avg)
        plt.legend(["loss", "loss_avg"])
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.xlim(0, len(X))
        plt.ylim(0, 1)
        
        rewards_avg = [sum(self.rewards[i:i+100])/100 for i in range(len(self.rewards)-100)]
        X = range(len(self.rewards))
        plt.subplot(1, 2, 2)
        plt.title("Rewards")
        plt.plot(X, self.rewards)
        plt.plot(X[100:], rewards_avg)
        plt.legend(["rewards", "rewards_avg"])
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.xlim(0, len(X))
        plt.ylim(-2, 1)

        plt.tight_layout()
        plt.show()

        fig.savefig(os.path.join(self.model_path,"results.png"))
