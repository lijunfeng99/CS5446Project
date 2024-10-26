import random
from utilis.agentbase import Agent

class RandomAgent(Agent):
    def __init__(self, name):
        super().__init__(name)  # 调用父类的构造函数

    def __call__(self, observation, configuration):
        # 使用不同的选择逻辑（示例）
        return random.choice([c for c in range(configuration.columns) if observation.board[c] == 0])