import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
from kaggle_environments import evaluate

from utilis.agentbase import Agent

class Double_Qlearning_agent_CNN_v2(Agent):
    def __init__(self, model_name= "Double_Qlearning_agent"):
        super().__init__(model_name)
        self.model_name = model_name
        self.memory = []
        self.epsilon = self.config.EPSILON_START

        # model
        self.model = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Flatten(),
            nn.Linear(64 * 7 * 6, 64),
            nn.Tanh(), 
            nn.Linear(64, 7) # Final layer with small std for output
        )

        self.target_model = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Flatten(),
            nn.Linear(64 * 7 * 6, 64),
            nn.Tanh(), 
            nn.Linear(64, 7)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.to(self.config.DEVICE)
        pass

    def convert_to_dual_channel(self, state):
        # 创建两个通道的状态：玩家1（1/0）和玩家2（1/0）
        state_reshaped = state.view(-1, 7, 6)  # 将每个样本恢复为 7x6 的棋盘
    
        # 对每个样本生成玩家1和玩家2的通道
        player1_channel = (state_reshaped == 1).int().float()  # 玩家1的位置填充1，其它为0
        player2_channel = (state_reshaped == 2).int().float()  # 玩家2的位置填充1，其它为0
        
        # 将两个通道堆叠为一个形状为 (batch_size, 2, 7, 6) 的张量
        dual_state = torch.stack([player1_channel, player2_channel], dim=1)
        
        return dual_state

    def forward(self, x):
        x = self.convert_to_dual_channel(x)
        return self.model(x)

    def target_forward(self, x):
        x = self.convert_to_dual_channel(x)
        return self.target_model(x)
    
    def __call__(self, observation, configuration):
        x = torch.tensor(observation["board"], dtype=torch.float32)
        x = x.unsqueeze(0)
        x = self.convert_to_dual_channel(x)
        x = x.to(self.config.DEVICE)
        q_values = self.forward(x)
        action = torch.argmax(q_values).item()
        return action
    
    def train(self):
        self.model.train()
        self.to(self.config.DEVICE)

        game_id = 0
        train_env = self.env.train(self.games[game_id])
        self.win = []
        total_reward = 0
        mx_reward = self.config.WIN_REWARD
        mi_reward = self.config.INVALID_MOVE_PENALTY

        tbar = tqdm(range(self.config.NUM_EPISODES))
        for episode in tbar:
            game_id = (game_id + 1) % len(self.games)
            train_env = self.env.train(self.games[game_id])
            
            if (episode + 1) % self.config.SAVE_CHECKPOINT_EVERY == 0:
                self.memory = [] # Clear memory after checkpoint
                self.save()
                self.to(self.config.DEVICE)

            state = train_env.reset()
            done = False
            total_reward = 0
            loss = 0
            
            while not done:
                action = self(state, self.env.configuration)

                # epsilon-greedy exploration
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.config.num_actions)

                next_state, reward, done, _ = train_env.step(action)
                if reward == None:
                    reward = self.config.INVALID_MOVE_PENALTY
                    self.win.append(0)
                if done:
                    if reward == 1:
                        reward = self.config.WIN_REWARD
                        self.win.append(1)
                    elif reward == -1:
                        reward = self.config.LOSE_PENALTY
                        self.win.append(0)
                    elif reward == 0:
                        reward = self.config.DRAW_PENALTY
                else:
                    reward = self.get_reward(board= state["board"], idx= self.action2index(action, board= state["board"]), label= state["mark"])
                
                # print(reward, (reward - mi_reward) / (mx_reward - mi_reward))
                # reward = (reward - mi_reward) / (mx_reward - mi_reward) # Normalize reward

                reward = self.rewardNormalizer.normalize(reward)
                total_reward = reward + total_reward * self.config.GAMMA

                # Store experience
                self.memory.append((state["board"], action, reward, next_state["board"], done))

                # Update state
                state = next_state
                
                # Sample a batch from memory and train the model
                if len(self.memory) > self.config.BATCH_SIZE:
                    loss = loss * 0.9 + self._replay() * 0.1
                
                # Clear memory if out of size
                if len(self.memory) > self.config.BUFFER_SIZE:
                    self._clear_memory()

            if episode % self.config.TARGET_UPDATE == 0:
                self.epsilon = max(self.epsilon * self.config.EPSILON_DECAY, self.config.EPSILON_END)

            tbar.set_postfix(Reward= round(np.mean(self.rewards[-100:]), 3), win= round(np.mean(self.win[-100:]), 3), loss= round(np.mean(self.loss[-100:]), 3))
            self.rewards.append(total_reward)
            self.loss.append(loss)

            if (episode + 1) % self.config.TARGET_UPDATE == 0:
                self._update_target_network()
        self.memory = [] # Clear memory after training

    def _update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _clear_memory(self):
        n = len(self.memory)
        probabilities = 0.8 * (1 - np.arange(n) / n)
        random_values = np.random.rand(n)
        mask = random_values > probabilities
        self.memory = [element for (i, element) in enumerate(self.memory) if mask[i]]

    def _replay(self):
        # Sample a batch of experiences
        batch = np.random.choice(len(self.memory), self.config.BATCH_SIZE, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

        states = torch.tensor(states, dtype=torch.float32).to(self.config.DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.config.DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.config.DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(self.config.DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.config.DEVICE)

        # Compute Q values
        q_values = self.forward(states)
        next_q_values = self.target_forward(next_states) # Double Q-learning

        target = rewards + (self.config.GAMMA * next_q_values.max(1)[0] * (1 - dones))
        expected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute loss
        loss = self.criterion(expected, target.detach())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_torch_model(self, path):
        torch.save(self.model.cpu().state_dict(), path + "model.pth")
        torch.save(self.target_model.cpu().state_dict(), path + "target_model.pth")
    
    def load_torch_model(self, path):
        self.model.load_state_dict(torch.load(path + "model.pth"))
        self.target_model.load_state_dict(torch.load(path + "target_model.pth"))
        self.to(self.config.DEVICE)


class Double_Qlearning_agent_CNN(Agent):
    def __init__(self, model_name= "Double_Qlearning_agent"):
        super().__init__(model_name)
        self.model_name = model_name
        self.memory = []
        self.epsilon = self.config.EPSILON_START

        # model
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Flatten(),
            nn.Linear(64 * 7 * 6, 64),
            nn.Tanh(), 
            nn.Linear(64, 7) # Final layer with small std for output
        )

        self.target_model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Flatten(),
            nn.Linear(64 * 7 * 6, 64),
            nn.Tanh(), 
            nn.Linear(64, 7)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.to(self.config.DEVICE)
        pass

    def forward(self, x):
        x = x.view(x.size(0), 1, 7, 6)
        return self.model(x)

    def target_forward(self, x):
        x = x.view(x.size(0), 1, 7, 6)
        return self.target_model(x)
    
    def __call__(self, observation, configuration):
        x = torch.tensor(observation["board"], dtype=torch.float32)
        x = x.unsqueeze(0)
        x = x.to(self.config.DEVICE)
        q_values = self.forward(x)
        action = torch.argmax(q_values).item()
        return action
    
    def train(self):
        self.model.train()
        self.to(self.config.DEVICE)

        game_id = 0
        train_env = self.env.train(self.games[game_id])
        self.win = []
        total_reward = 0
        mx_reward = self.config.WIN_REWARD
        mi_reward = self.config.INVALID_MOVE_PENALTY

        tbar = tqdm(range(self.config.NUM_EPISODES))
        for episode in tbar:
            game_id = (game_id + 1) % len(self.games)
            train_env = self.env.train(self.games[game_id])
            
            if (episode + 1) % self.config.SAVE_CHECKPOINT_EVERY == 0:
                self.memory = [] # Clear memory after checkpoint
                self.save()
                self.to(self.config.DEVICE)

            state = train_env.reset()
            done = False
            total_reward = 0
            loss = 0
            
            while not done:
                action = self(state, self.env.configuration)

                # epsilon-greedy exploration
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.config.num_actions)

                next_state, reward, done, _ = train_env.step(action)
                if reward == None:
                    reward = -5
                if done:
                    if reward == 1:
                        self.win.append(1)
                        reward = 5
                    else:
                        self.win.append(0)
                        reward = -5
                else:
                    reward = -0.03 # self.calculate_reward(state["board"], action, state["mark"])
                # if reward == None:
                #     reward = self.config.INVALID_MOVE_PENALTY
                # if done:
                #     if reward == 1:
                #         reward = self.config.WIN_REWARD
                #         self.win.append(1)
                #     elif reward == -1:
                #         reward = self.config.LOSE_PENALTY
                #         self.win.append(0)
                #     elif reward == 0:
                #         reward = self.config.DRAW_PENALTY
                # else:
                #     reward = -0.02
                    # reward = self.get_reward(board= state["board"], idx= self.action2index(action, board= state["board"]), label= state["mark"])
                
                # print(reward, (reward - mi_reward) / (mx_reward - mi_reward))
                # reward = (reward - mi_reward) / (mx_reward - mi_reward) # Normalize reward

                reward = self.rewardNormalizer.normalize(reward)
                total_reward = reward + total_reward * self.config.GAMMA

                # Store experience
                self.memory.append((state["board"], action, reward, next_state["board"], done))

                # Update state
                state = next_state
                
                # Sample a batch from memory and train the model
                if len(self.memory) > self.config.BATCH_SIZE:
                    loss = loss * 0.9 + self._replay() * 0.1
                
                # Clear memory if out of size
                if len(self.memory) > self.config.BUFFER_SIZE:
                    self._clear_memory()

            if episode % self.config.TARGET_UPDATE == 0:
                self.epsilon = max(self.epsilon * self.config.EPSILON_DECAY, self.config.EPSILON_END)

            tbar.set_postfix(Reward= round(np.mean(self.rewards[-100:]), 3), win= round(np.mean(self.win[-100:]), 3), loss= round(np.mean(self.loss[-100:]), 3))
            self.rewards.append(total_reward)
            self.loss.append(loss)

            if (episode + 1) % self.config.TARGET_UPDATE == 0:
                self._update_target_network()
        self.memory = [] # Clear memory after training

    def _update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _clear_memory(self):
        n = len(self.memory)
        probabilities = 0.8 * (1 - np.arange(n) / n)
        random_values = np.random.rand(n)
        mask = random_values > probabilities
        self.memory = [element for (i, element) in enumerate(self.memory) if mask[i]]

    def _replay(self):
        # Sample a batch of experiences
        batch = np.random.choice(len(self.memory), self.config.BATCH_SIZE, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

        states = torch.tensor(states, dtype=torch.float32).to(self.config.DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.config.DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.config.DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(self.config.DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.config.DEVICE)

        # Compute Q values
        q_values = self.forward(states)
        next_q_values = self.target_forward(next_states) # Double Q-learning

        target = rewards + (self.config.GAMMA * next_q_values.max(1)[0] * (1 - dones))
        expected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute loss
        loss = self.criterion(expected, target.detach())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_torch_model(self, path):
        torch.save(self.model.cpu().state_dict(), path + "model.pth")
        torch.save(self.target_model.cpu().state_dict(), path + "target_model.pth")
    
    def load_torch_model(self, path):
        self.model.load_state_dict(torch.load(path + "model.pth"))
        self.target_model.load_state_dict(torch.load(path + "target_model.pth"))
        self.to(self.config.DEVICE)


class Double_Qlearning_agent_self_play(Agent):
    def __init__(self, model_name= "Double_Qlearning_agent"):
        super().__init__(model_name)
        self.model_name = model_name
        self.memory = []
        self.epsilon = self.config.EPSILON_START

        # model
        self.model = nn.Sequential(
            nn.Linear(self.config.input_shape, 128),
            nn.ReLU(), 
            nn.Linear(128, 128),
            nn.ReLU(), 
            nn.Linear(128, 128),
            nn.ReLU(), 
            nn.Linear(128, 32),
            nn.Tanh(), 
            nn.Linear(32, self.config.num_actions)
        )

        self.target_model = nn.Sequential(
            nn.Linear(self.config.input_shape, 128),
            nn.ReLU(), 
            nn.Linear(128, 128),
            nn.ReLU(), 
            nn.Linear(128, 128),
            nn.ReLU(), 
            nn.Linear(128, 32),
            nn.Tanh(), 
            nn.Linear(32, self.config.num_actions)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.to(self.config.DEVICE)
        pass

    def forward(self, x):
        return self.model(x)

    def target_forward(self, x):
        return self.target_model(x)
    
    def __call__(self, observation, configuration):
        x = torch.tensor(observation["board"], dtype=torch.float32)
        x = x.unsqueeze(0)
        x = x.to(self.config.DEVICE)
        q_values = self.forward(x)
        action = torch.argmax(q_values).item()
        return action
    
    def mean_reward(self, rewards):
        return sum(r[0] if r[0] is not None else -1 for r in rewards) / float(len(rewards))

    def train(self):
        self.model.train()
        self.to(self.config.DEVICE)

        self.win = []
        total_steps = 0

        tbar = tqdm(range(self.config.NUM_EPISODES))
        for episode in tbar:
            
            if (episode + 1) % self.config.SAVE_CHECKPOINT_EVERY == 0:
                self.memory = [] # Clear memory after checkpoint
                self.save()
                self.to(self.config.DEVICE)

            one_play = self.env.run([self, self])
            loss = 0

            for step in range(len(one_play)):

                action = one_play[step][0]["action"]
                state = one_play[step - 1][0]["observation"]["board"]
                next_state = one_play[step][0]["observation"]["board"]
                done = one_play[step][0]["status"] == "DONE"
                reward = one_play[step][0]["reward"]
                if reward == None:
                    reward = -1
                # idx = self.action2index(action, board= state)
                # mark = next_state[idx]
                # reward = self.calculate_reward(board= state, action= action, player_id= mark)
                total_steps += 1

                # reservior sampling
                if len(self.memory) < self.config.BUFFER_SIZE:
                    self.memory.append((state, action, reward, next_state, done))
                else:
                    if np.random.rand() < 1.0 * self.config.BUFFER_SIZE / total_steps:
                        self.memory[np.random.randint(0, self.config.BUFFER_SIZE)] = (state, action, reward, next_state, done)
                
                if len(self.memory) > self.config.BATCH_SIZE:
                    loss = loss * 0.9 + self._replay() * 0.1

            tbar.set_postfix(Reward= round(np.mean(self.rewards[-100:]), 3), win= round(np.mean(self.win[-100:]), 3), loss= round(np.mean(self.loss[-100:]), 3), steps= len(one_play))
            self.rewards.append(self.mean_reward(evaluate("connectx", [self, "random"], num_episodes=1) + evaluate("connectx", ["random", self], num_episodes=1)))
            self.loss.append(loss)
            self.win.append((self.rewards[-1] + 1) / 2)

            if (episode + 1) % self.config.TARGET_UPDATE == 0:
                self.memory = self.memory[:int(len(self.memory) * 0.8)] # Clear memory after checkpoint
                self._update_target_network()
        self.memory = [] # Clear memory after training

    def _update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _clear_memory(self):
        # 只保留最后一部分的经验
        n = len(self.memory)
        probabilities = 0.8 * (1 - np.arange(n) / n)
        random_values = np.random.rand(n)
        mask = random_values > probabilities
        self.memory = [element for (i, element) in enumerate(self.memory) if mask[i]]

    def _replay(self):
        # Sample a batch of experiences
        batch = np.random.choice(len(self.memory), self.config.BATCH_SIZE, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

        states = torch.tensor(states, dtype=torch.float32).to(self.config.DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.config.DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.config.DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(self.config.DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.config.DEVICE)

        # Compute Q values
        q_values = self.forward(states)
        next_q_values = self.target_forward(next_states) # Double Q-learning

        target = rewards + (self.config.GAMMA * next_q_values.max(1)[0] * (1 - dones))
        expected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute loss
        loss = self.criterion(expected, target.detach())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_torch_model(self, path):
        torch.save(self.model.cpu().state_dict(), path + "model.pth")
        torch.save(self.target_model.cpu().state_dict(), path + "target_model.pth")
    
    def load_torch_model(self, path):
        self.model.load_state_dict(torch.load(path + "model.pth"))
        self.target_model.load_state_dict(torch.load(path + "target_model.pth"))
        self.to(self.config.DEVICE)

class Double_Qlearning_agent(Agent):
    def __init__(self, model_name= "Double_Qlearning_agent"):
        super().__init__(model_name)
        self.model_name = model_name
        self.memory = []
        self.epsilon = self.config.EPSILON_START

        # model
        self.model = nn.Sequential(
            nn.Linear(self.config.input_shape, 128),
            nn.ReLU(), 
            nn.Linear(128, 128),
            nn.ReLU(), 
            nn.Linear(128, 128),
            nn.Tanh(), 
            nn.Linear(128, self.config.num_actions)
        )

        self.target_model = nn.Sequential(
            nn.Linear(self.config.input_shape, 128),
            nn.ReLU(), 
            nn.Linear(128, 128),
            nn.ReLU(), 
            nn.Linear(128, 128),
            nn.Tanh(), 
            nn.Linear(128, self.config.num_actions)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.to(self.config.DEVICE)
        pass

    def forward(self, x):
        return self.model(x)

    def target_forward(self, x):
        return self.target_model(x)
    
    def __call__(self, observation, configuration):
        x = torch.tensor(observation["board"], dtype=torch.float32)
        x = x.unsqueeze(0)
        x = x.to(self.config.DEVICE)
        q_values = self.forward(x)
        action = torch.argmax(q_values).item()
        return action
    
    def train(self):
        self.model.train()
        self.to(self.config.DEVICE)

        game_id = 0
        train_env = self.env.train(self.games[game_id])
        self.win = []
        total_reward = 0
        mx_reward = self.config.WIN_REWARD
        mi_reward = self.config.INVALID_MOVE_PENALTY

        tbar = tqdm(range(self.config.NUM_EPISODES))
        for episode in tbar:
            game_id = (game_id + 1) % len(self.games)
            train_env = self.env.train(self.games[game_id])
            
            if (episode + 1) % self.config.SAVE_CHECKPOINT_EVERY == 0:
                self.memory = [] # Clear memory after checkpoint
                self.save()
                self.to(self.config.DEVICE)

            state = train_env.reset()
            done = False
            total_reward = 0
            loss = 0
            
            while not done:
                action = self(state, self.env.configuration)

                # epsilon-greedy exploration
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.config.num_actions)

                next_state, reward, done, _ = train_env.step(action)
                if reward == None:
                    reward = -1
                if done:
                    if reward == 1:
                        self.win.append(1)
                        reward = 100
                    else:
                        self.win.append(0)
                        reward = -100
                else:
                    reward = self.calculate_reward(state["board"], action, state["mark"])
                # if reward == None:
                #     reward = self.config.INVALID_MOVE_PENALTY
                # if done:
                #     if reward == 1:
                #         reward = self.config.WIN_REWARD
                #         self.win.append(1)
                #     elif reward == -1:
                #         reward = self.config.LOSE_PENALTY
                #         self.win.append(0)
                #     elif reward == 0:
                #         reward = self.config.DRAW_PENALTY
                # else:
                #     reward = -0.02
                    # reward = self.get_reward(board= state["board"], idx= self.action2index(action, board= state["board"]), label= state["mark"])
                
                # print(reward, (reward - mi_reward) / (mx_reward - mi_reward))
                # reward = (reward - mi_reward) / (mx_reward - mi_reward) # Normalize reward

                reward = self.rewardNormalizer.normalize(reward)
                total_reward = reward + total_reward * self.config.GAMMA

                # Store experience
                self.memory.append((state["board"], action, reward, next_state["board"], done))

                # Update state
                state = next_state
                
                # Sample a batch from memory and train the model
                if len(self.memory) > self.config.BATCH_SIZE:
                    loss = loss * 0.9 + self._replay() * 0.1
                
                # Clear memory if out of size
                if len(self.memory) > self.config.BUFFER_SIZE:
                    self._clear_memory()

            if episode % self.config.TARGET_UPDATE == 0:
                self.epsilon = max(self.epsilon * self.config.EPSILON_DECAY, self.config.EPSILON_END)

            tbar.set_postfix(Reward= round(np.mean(self.rewards[-100:]), 3), win= round(np.mean(self.win[-100:]), 3), loss= round(np.mean(self.loss[-100:]), 3))
            self.rewards.append(total_reward)
            self.loss.append(loss)

            if (episode + 1) % self.config.TARGET_UPDATE == 0:
                self._update_target_network()
        self.memory = [] # Clear memory after training

    def _update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _clear_memory(self):
        # 只保留最后一部分的经验
        n = len(self.memory)
        probabilities = 0.8 * (1 - np.arange(n) / n)
        random_values = np.random.rand(n)
        mask = random_values > probabilities
        self.memory = [element for (i, element) in enumerate(self.memory) if mask[i]]

    def _replay(self):
        # Sample a batch of experiences
        batch = np.random.choice(len(self.memory), self.config.BATCH_SIZE, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

        states = torch.tensor(states, dtype=torch.float32).to(self.config.DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.config.DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.config.DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(self.config.DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.config.DEVICE)

        # Compute Q values
        q_values = self.forward(states)
        next_q_values = self.target_forward(next_states) # Double Q-learning

        target = rewards + (self.config.GAMMA * next_q_values.max(1)[0] * (1 - dones))
        expected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute loss
        loss = self.criterion(expected, target.detach())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_torch_model(self, path):
        torch.save(self.model.cpu().state_dict(), path + "model.pth")
        torch.save(self.target_model.cpu().state_dict(), path + "target_model.pth")
    
    def load_torch_model(self, path):
        self.model.load_state_dict(torch.load(path + "model.pth"))
        self.target_model.load_state_dict(torch.load(path + "target_model.pth"))
        self.to(self.config.DEVICE)