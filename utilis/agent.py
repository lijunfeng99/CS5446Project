from utilis.Random_agent import RandomAgent
from utilis.Naive_Qlearning_agent import Naive_Qlearning_agent
from utilis.Double_Qlearning_agent import Double_Qlearning_agent
from utilis.Monte_Carlo_learning import MonteCarloAgent as Monte_Carlo_agent
from utilis.TD3_agent import TD3_agent

class RandomAgent(RandomAgent):
    def __init__(self, name):
        super().__init__(name)  # 调用父类的构造函数

class Naive_Qlearning_agent(Naive_Qlearning_agent):
    def __init__(self, name):
        super().__init__(name)  # 调用父类的构造函数

class Double_Qlearning_agent(Double_Qlearning_agent):
    def __init__(self, name):
        super().__init__(name)  # 调用父类的构造函数

class Monte_Carlo_agent(Monte_Carlo_agent):
    def __init__(self, name):
        super().__init__(name)  # 调用父类的构造函数

class TD3_agent(TD3_agent):
    def __init__(self, name):
        super().__init__(name)