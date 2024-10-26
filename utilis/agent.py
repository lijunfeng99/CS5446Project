from utilis.Random_agent import RandomAgent
from utilis.Naive_Qlearning_agent import Naive_Qlearning_agent

class RandomAgent(RandomAgent):
    def __init__(self, name):
        super().__init__(name)  # 调用父类的构造函数

class Naive_Qlearning_agent(Naive_Qlearning_agent):
    def __init__(self, name):
        super().__init__(name)  # 调用父类的构造函数