import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utilis.agentbase import Agent

class Double_Qlearning_agent(Agent):
    def __init__(self, model_name= "Double_Qlearning_agent"):
        super().__init__(model_name)
        self.model_name = model_name
        self.memory = []
        self.epsilon = self.config.EPSILON_START

        # model
        self.model = nn.Sequential(
            nn.Linear(self.config.input_shape, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, self.config.num_actions)
        )

        self.target_model = nn.Sequential(
            nn.Linear(self.config.input_shape, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(), nn.Dropout(0.2),
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
        x = torch.tensor(observation.board, dtype=torch.float32)
        x = x.unsqueeze(0)
        x = x.to(self.config.DEVICE)
        q_values = self.forward(x)
        action = torch.argmax(q_values).item()
        return action
    
    def train(self):
        self.model.train()
        self.to(self.config.DEVICE)
        train_env = self.env.train([None, "random"]) # random, negamax
        win = 0
        sum_loss = 0
        sum_reward = 0

        tbar = tqdm(range(self.config.NUM_EPISODES))
        for episode in tbar:
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
                    reward = self.config.INVALID_MOVE_PENALTY # Invalid move penalty
                total_reward += reward

                # Store experience
                self.memory.append((state.board, action, reward, next_state.board, done))
                if reward == 1: # Since the positive sample is less, we give it a higher weight
                    self.memory.append((state.board, action, reward, next_state.board, done))
                    self.memory.append((state.board, action, reward, next_state.board, done))
                    self.memory.append((state.board, action, reward, next_state.board, done))
                    self.memory.append((state.board, action, reward, next_state.board, done))
                    self.memory.append((state.board, action, reward, next_state.board, done))

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
            win += 1 if total_reward > 0 else 0

            sum_loss += loss
            sum_reward += total_reward
            tbar.set_postfix(Reward= round(sum_reward / (episode + 1), 3), win= round(win / (episode + 1), 3), loss= round(sum_loss / (episode + 1), 3))
            self.rewards.append(total_reward)
            self.loss.append(loss)

            if (episode + 1) % self.config.TARGET_UPDATE == 0:
                self._update_target_network()
        self.memory = [] # Clear memory after training

    def _update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _clear_memory(self):
        # 只保留最后一部分的经验
        self.memory = self.memory[-int(self.config.BUFFER_SIZE * 0.8):]

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