import argparse
from collections import namedtuple
from itertools import count

import os, sys, random
import numpy as np
from utilis.agentbase import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import gym
directory = './model/TD3/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()



# Set seeds
# env.seed(args.seed)
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
min_Val = torch.tensor(1e-7).float().to(device) # min value


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []
        # print(self.storage[0])
        for i in ind:
            X, Y, U, R, D = self.storage[i]
            X = dict(X)
            Y = dict(Y)
            # print(X['board'])
            # print(Y['board'])
            x.append(np.array(X['board'], copy=False))
            y.append(np.array(Y['board'], copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # # 卷积层用于提取棋盘特征
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)

        # # 全连接层用于决策输出
        # self.fc1 = nn.Linear(64 * state_dim, 400)
        self.fc1 = nn.Linear(state_dim*6, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        # 棋盘输入可能是1D，需要转换为2D矩阵（例如6x7的ConnectX）
        state = state.view(-1, 1, 6, 7)  # 根据实际棋盘大小调整形状
        a = F.relu(self.conv1(state))
        a = F.relu(self.conv2(a))
        
        # a = F.relu(self.conv3(a))
        a = a.view(a.size(0), -1)  # 展平以输入全连接层
        # print(a.size())
        a = F.relu(self.fc1(a))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # 卷积层用于提取棋盘状态特征
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)

        # # 全连接层用于将卷积特征和动作拼接后的特征映射到 Q 值
        self.fc1 = nn.Linear(6 * state_dim + action_dim, 400)
        # self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        # 将棋盘状态调整为 2D 矩阵（假设 ConnectX 棋盘大小为 6x7）
        state = state.view(-1, 1, 6, 7)  # 根据棋盘实际尺寸进行调整

        # # 卷积层提取特征
        q = F.relu(self.conv1(state))
        q = F.relu(self.conv2(q))
        
        # q = F.relu(self.conv3(q))
        q = q.view(q.size(0), -1)  # 展平卷积特征
        
        # 将卷积特征和动作拼接
        q = torch.cat([q, action], dim=1)
        # print(q.size(), action.size())
        # 全连接层进一步处理
        q = F.relu(self.fc1(q))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q

# class Actor(nn.Module):

#     def __init__(self, state_dim, action_dim, max_action):
#         super(Actor, self).__init__()

#         self.fc1 = nn.Linear(state_dim, 400)
#         self.fc2 = nn.Linear(400, 300)
#         self.fc3 = nn.Linear(300, action_dim)

#         self.max_action = max_action

#     def forward(self, state):
#         a = F.relu(self.fc1(state))
#         a = F.relu(self.fc2(a))
#         a = torch.tanh(self.fc3(a)) * self.max_action
#         return a


# class Critic(nn.Module):

#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()

#         self.fc1 = nn.Linear(state_dim + action_dim, 400)
#         self.fc2 = nn.Linear(400, 300)
#         self.fc3 = nn.Linear(300, 1)

#     def forward(self, state, action):
#         state_action = torch.cat([state, action], 1)

#         q = F.relu(self.fc1(state_action))
#         q = F.relu(self.fc2(q))
#         q = self.fc3(q)
#         return q

    
class TD3():
    def __init__(self, state_dim, action_dim, max_action, config):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=config.LEARNING_RATE)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(),lr=config.LEARNING_RATE)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(),lr=config.LEARNING_RATE)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer(config.BUFFER_SIZE)
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.config = config

    def select_action(self, state):
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, num_iteration):

        if self.num_training % 500 == 0:
            print("====================================")
            print("model has been trained for {} times...".format(self.num_training))
            print("====================================")
        for i in range(num_iteration):
            x, y, u, r, d = self.memory.sample(self.config.BATCH_SIZE)
            # print(y)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, self.config.policy_noise).to(device)
            noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.config.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)
            
            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            # print(loss_Q1, loss_Q2)
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)
            # Delayed policy updates:
            if i % self.config.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1- self.config.tau) * target_param.data) + self.config.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - self.config.tau) * target_param.data) + self.config.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - self.config.tau) * target_param.data) + self.config.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory+'actor.pth')
        torch.save(self.actor_target.state_dict(), directory+'actor_target.pth')
        torch.save(self.critic_1.state_dict(), directory+'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), directory+'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), directory+'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), directory+'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


class TD3_agent(Agent):
    def __init__(self, model_name = 'TD3'):
        super().__init__(model_name)
        self.model_name = model_name
        self.agent = TD3(self.config.input_shape, self.config.num_actions, 1, self.config)

    def train(self):
        """Train the agent in ConnectX environment as both first and second player."""
        print("====================================")
        print("Starting Training with Alternating First and Second Player Roles...")
        print("====================================")
        
        # 加载模型（如配置中指定）
        if self.config.LOAD:
            self.agent.load()
        # env = self.env.train(["random", None])  # 代理为后手
        # player_role = "second player"
        # 设置训练环境
        # env = self.env.train([None, "random"])  
        for episode in range(self.config.num_iteration):
            # 每次迭代，交替设置先手和后手
            if episode % 2 == 0:
                env = self.env.train([None, "random"])  # 代理为先手
                player_role = "first player"
            else:
                env = self.env.train(["random", None])  # 代理为后手
                player_role = "second player"
            
            # 初始化变量
            # 代理为先手
            # player_role = "first player"
            state = env.reset()
            episode_reward = 0

            for t in range(self.config.max_episode):
                # 选择并执行动作
                action, action_to_update = self(state, self.env.configuration)
                next_state, reward, done, info = env.step(action)

                # 若无奖励（即非法操作），使用配置中的罚分
                if reward is None:
                    reward = self.config.INVALID_MOVE_PENALTY
                
                episode_reward += reward
                
                # 存储至记忆库
                self.agent.memory.push((state, next_state, action_to_update, reward, float(done)))
                
                # 达到指定记忆大小后，执行模型更新
                if len(self.agent.memory.storage) >= self.config.BUFFER_SIZE:
                    self.agent.update(10)

                # 更新状态
                state = next_state
                
                # 终止条件：回合结束或达到最大回合数
                if done or t == self.config.max_episode - 1:
                    # 记录每回合总奖励
                    self.agent.writer.add_scalar('Episode Reward', episode_reward, global_step=episode)
                    if episode % self.config.print_log == 0:
                        print(f"Episode {episode} as {player_role}, Reward: {episode_reward:.2f}, Steps: {t}")
                    episode_reward = 0
                    break

            # 定期保存模型
            if episode % self.config.log_interval == 0:
                self.agent.save()
    
        print("Training complete.")

    def load(self):
        self.agent.load()
        
    def test(self):
        ep_r = 0
        self.agent.load()
        env = self.env.train([None, "random"])
        for i in range(self.config.iteration):
            state = env.reset()
            for t in count():
                action = self(state, self.env.configuration)
                next_state, reward, done, info = env.step(np.float32(action))
                print(self.env.configuration)
                print(next_state, reward, done, info, action)
                if reward == None:
                    reward = self.config.INVALID_MOVE_PENALTY
                ep_r += reward
                if not env.action_space.contains(action):
                    print("Invalid action:", action)
                if done or t ==2000 :
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    break
                state = next_state


    def __call__(self, observation, configuration):
        observation = dict(observation)
        x = torch.tensor(observation['board'], dtype=torch.float32)
        x = x.unsqueeze(0)
        x = x.to(device)
        action_to_update = self.agent.select_action(x)
        action = action_to_update + np.random.normal(0, self.config.exploration_noise, size=self.config.num_actions)
        action = np.argmax(action).item()
        return action, action_to_update
    
