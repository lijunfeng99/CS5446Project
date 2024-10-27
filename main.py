from utilis.agent import *
from utilis.evaluate import cross_validation, draw_result

if __name__ == '__main__':
    agents_names = ['RandomAgent', 'Q-learning-v1', 'DDQN_v1', 'Monte-Carlo-agent-v1']
    agents = {
        "RandomAgent": RandomAgent("RandomAgent"),
        "Q-learning-v1": Naive_Qlearning_agent("Q-learning-v1"),
        "DDQN_v1": Double_Qlearning_agent("DDQN_v1"),
        "Monte-Carlo-agent-v1": Monte_Carlo_agent("Monte-Carlo-agent-v1")
    }
    
    for k, v in agents.items():
        print(f"Agent {k} is ready to play.")
        v.load()
    
    results = cross_validation(agents)
    results.to_csv('results.csv', index=False)
    draw_result(results)