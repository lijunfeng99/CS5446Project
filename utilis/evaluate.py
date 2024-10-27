from kaggle_environments import evaluate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def draw_result(df):
    agents = df['agent1'].unique()

    reward_matrix = pd.DataFrame(index=agents, columns=agents, data=0)

    # 填充奖励矩阵
    for _, row in df.iterrows():
        reward_matrix.loc[row['agent1'], row['agent2']] = row['mean_reward']

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(reward_matrix.astype(float), annot=True, cmap='coolwarm', center=0)
    plt.title('Mean Reward Matrix')
    plt.xlabel('Agent 2')
    plt.ylabel('Agent 1')
    plt.show()
    fig.savefig('results.png')
