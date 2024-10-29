from utilis.create_submision import write_to_submission
from submission import my_agent
from kaggle_environments.utils import Struct

if __name__ == "__main__":
    write_to_submission("DDQN_v1")
    observation = {'remainingOverageTime': 60, 'step': 24, 'board': [0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 2, 1, 2, 0, 2, 0, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1], 'mark': 1}
    configuration = {'episodeSteps': 1000, 'actTimeout': 2, 'runTimeout': 1200, 'columns': 7, 'rows': 6, 'inarow': 4, 'agentTimeout': 60, 'timeout': 2}
    action = my_agent(Struct(**observation), Struct(**configuration))
    print(action)
    pass