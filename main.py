from utilis.agent import *
from kaggle_environments import evaluate
import pandas as pd

def mean_reward(rewards):
    return sum(r[0] if r[0] is not None else -1 for r in rewards) / float(len(rewards))

def cross_validation(agents, num_episodes=100):
    res = pd.DataFrame(columns=['agent1', 'agent2','mean_reward'])
    for agent_name1, agent1 in agents.items():
        for agent_name2, agent2 in agents.items():
            if agent_name1 == agent_name2:
                continue
            results = evaluate("connectx", [agent1, agent2], num_episodes=num_episodes)
            mean_reward_value = mean_reward(results)
            print(f"Agent {agent_name1} vs agent {agent_name2} mean reward: {mean_reward_value}")
            res.loc[len(res)] = [agent_name1, agent_name2, mean_reward_value]
    return res

if __name__ == '__main__':
    agents_names = ['RandomAgent', 'Q-learning-v1', 'DDQN_v1']
    agents = {
        "RandomAgent": RandomAgent("RandomAgent"),
        "Q-learning-v1": Naive_Qlearning_agent("Q-learning-v1"),
        "DDQN_v1": Double_Qlearning_agent("DDQN_v1")
    }
    
    for k, v in agents.items():
        print(f"Agent {k} is ready to play.")
        v.load()
    
    results = cross_validation(agents)
    results.to_csv('results.csv', index=False)