from kaggle_environments import evaluate, make
# from utilis.agent import RandomAgent as TrainableAgent
# from utilis.agent import Naive_Qlearning_agent as TrainableAgent
from utilis.agent import Double_Qlearning_agent as TrainableAgent
# from utilis.agent import Monte_Carlo_agent as TrainableAgent
from utilis.config import Config

def mean_reward(rewards):
    return sum(r[0] if r[0] is not None else -1 for r in rewards) / float(len(rewards))

if __name__ == "__main__":
    # 1. Create an environment
    env = make("connectx", debug=True)
    env.reset()

    # 2. Define your agents configuratio
    model_name = "DDQN_v1"
    # config = Config(model_name)
    # config.save()

    # 3. Create your agent and train it
    my_agent = TrainableAgent(model_name)
    # my_agent.save()
    my_agent.load()
    my_agent.model.eval()
    print(my_agent.test())
    # my_agent.train()
    # my_agent.save()
    # my_agent.draw_training_records()

    # 4. Evaluate your agent
    env.reset()
    my_agent.load()
    my_agent.model.eval()

    # Run multiple episodes to estimate its performance.
    print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))
    print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))