import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.distributions import Categorical

from utilis.agentbase import Agent


class A2C_agent(Agent):
    def __init__(self, model_name="A2C_agent"):
        super().__init__(model_name)
        self.model_name = model_name
        self.memory = []
        self.epsilon = self.config.EPSILON_START

        # model
        self.critic = nn.Sequential(
            nn.Linear(self.config.input_shape, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.model = nn.Sequential(
            nn.Linear(self.config.input_shape, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, self.config.num_actions),
            nn.Softmax()
        )

        self.optimizer =optim.Adam(list(self.model.parameters()) + list(self.critic.parameters()), lr=self.config.LEARNING_RATE)

        self.criterion = nn.MSELoss()
        self.to(self.config.DEVICE)
        pass

    def forward(self, x):
        value = self.critic(x)
        probs = self.model(x)
        dist = Categorical(probs)
        # print("value", value)
        # print("probs", probs)
        return dist, value

    def target_forward(self, x):
        return self.target_model(x)

    def __call__(self, observation, configuration):
        # print("observation")
        # print(observation)
        # print("len(observation)", len(observation.board))

        state = torch.FloatTensor(observation.board).to("cuda")
        dist, value = self.forward(state)

        # action = dist.sample()
        # x = torch.tensor(observation.board, dtype=torch.float32)
        # x = x.unsqueeze(0)
        # x = x.to(self.config.DEVICE)
        # q_values = self.forward(x)
        # action = torch.argmax(q_values).item()

        return dist, value

    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        # print(rewards)
        # print("masks", masks)
        for step in reversed(range(len(rewards))):
            if rewards[step] is None:
                rewards[step] = 0
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def train(self):
        self.model.train()
        self.critic.train()
        self.to(self.config.DEVICE)
        train_env = self.env.train([None, "random"])  # random, negamax
        win = 0
        sum_loss = 0
        sum_reward = 0



        tbar = tqdm(range(self.config.NUM_EPISODES))
        for episode in tbar:
            state = train_env.reset()
            done = False
            total_reward = 0
            loss = 0
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0
            while not done:
                # print(state)
                # state = torch.FloatTensor(state.board).to('cuda')
                # print(state.shape)
                # dist, value = self.forward(state)

                # action = dist.sample()
                dist, value = self(state, self.env.configuration)
                action = dist.sample()
                # action = np.random.choice(self.config.num_actions)
                # print("numpy action ", np.random.choice(self.config.num_actions))
                # print("type:", type(np.random.choice(self.config.num_actions)))
                # action = self(state, self.env.configuration)
                next_state, reward, done, _ = train_env.step(int(action))
                if reward is None:
                    reward = self.config.INVALID_MOVE_PENALTY # Invalid move penalty
                total_reward += reward
                # print(next_state)
                # print(reward)
                # print(done)
                # # print(action.cpu().numpy())
                # print(action)
                # print(action.shape)
                # print(type(int(action.cpu())))
                # print(0 / 0)

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                # print("reward", reward)
                rewards.append(reward)
                masks.append(1 if not done else 0)

                state = next_state
            next_state = torch.FloatTensor(next_state.board).to('cuda')
            _, next_value = self.forward(next_state)
            returns = self.compute_returns(next_value, rewards, masks)
            # print("log_probs", log_probs)
            log_probs = torch.tensor(log_probs).to("cuda")
            returns = torch.cat(returns).detach()
            values = torch.cat(values)

            advantage = returns - values

            model_loss = -(log_probs * advantage).mean()
            critic_loss = advantage.pow(2).mean()

            loss = model_loss + 0.5 * critic_loss - 0.001 * entropy
            # loss = loss.to("cpu")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss = loss.detach().to("cpu")
            self.loss.append(loss)
            self.rewards.append(total_reward)

