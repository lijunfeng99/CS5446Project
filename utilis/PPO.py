import torch
import torch.nn as nn
from utilis.agentbase import Agent
import numpy as np
from tqdm import tqdm
import random
import torch.nn.functional as F
import torch.optim as optim
from multiprocessing import Pool
import copy

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize the weights and biases of a layer.

    Args:
        layer (nn.Module): The layer to initialize.
        std (float): Standard deviation for orthogonal initialization.
        bias_const (float): Constant value for bias initialization.

    Returns:
        nn.Module: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)  # Orthogonal initialization
    torch.nn.init.constant_(layer.bias, bias_const)  # Constant bias
    return layer

class PPO_agent(Agent):
    def __init__(self, model_name= "PPO"):
        """Initialize the Actor-Critic agent with actor and critic networks."""
        super().__init__(model_name)

        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_objective_history = []

        self.reward_history = []
        self.episode_history = []

        ### ------------- TASK 1.1 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        actor_input_dim = self.config.input_shape  # Input dimension for the actor
        actor_output_dim = self.config.num_actions  # Output dimension for the actor (number of actions)
        critic_input_dim = self.config.input_shape  # Input dimension for the critic
        critic_output_dim = 1  # Output dimension for the critic (value estimate)
        ### ------ YOUR CODES END HERE ------- ###

        # Define the actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(actor_input_dim, 64)),
            nn.Tanh(), 
            layer_init(nn.Linear(64, 128)),
            nn.Tanh(), 
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(), 
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(), 
            layer_init(nn.Linear(64, 32)),
            nn.Tanh(), 
            layer_init(nn.Linear(32, actor_output_dim), std=0.01),  # Final layer with small std for output
        )

        # Define the critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_input_dim, 64)),
            nn.Tanh(), 
            layer_init(nn.Linear(64, 128)),
            nn.Tanh(), 
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(), 
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(), 
            layer_init(nn.Linear(64, 32)),
            nn.Tanh(), 
            layer_init(nn.Linear(32, 8)),
            nn.Tanh(), 
            layer_init(nn.Linear(8, critic_output_dim), std=1.0),  # Standard output layer for value
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)

        self.to(self.config.DEVICE)

    def get_value(self, x):
        """Calculate the estimated value for a given state.

        Args:
            x (torch.Tensor): Input state, shape: (batch_size, observation_size)

        Returns:
            torch.Tensor: Estimated value for the state, shape: (batch_size, 1)
        """
        ### ------------- TASK 1.2 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        value = self.critic(x)  # Forward pass through the critic network
        ### ------ YOUR CODES END HERE ------- ###
        return value

    def get_probs(self, x):
        """Calculate the action probabilities for a given state.

        Args:
            x (torch.Tensor): Input state, shape: (batch_size, observation_size)

        Returns:
            torch.distributions.Categorical: Categorical distribution over actions.
        """
        ### ------------- TASK 1.3 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        logits = self.actor(x)  # Get logits from the actor network
        probs = torch.distributions.Categorical(logits=logits)  # Create a categorical distribution from the logits
        ### ------ YOUR CODES END HERE ------- ###
        return probs

    def get_action(self, probs):
        """Sample an action from the action probabilities.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.

        Returns:
            torch.Tensor: Sampled action, shape: (batch_size, 1)
        """
        ### ------------- TASK 1.4 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        action = probs.sample()  # Sample an action based on the probabilities
        ### ------ YOUR CODES END HERE ------- ###
        return action

    def get_action_logprob(self, probs, action):
        """Compute the log probability of a given action.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.
            action (torch.Tensor): Selected action, shape: (batch_size, 1)

        Returns:
            torch.Tensor: Log probability of the action, shape: (batch_size, 1)
        """
        ### ------------- TASK 1.5 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        logprob = probs.log_prob(action)  # Calculate log probability of the sampled action
        ### ------ YOUR CODES END HERE ------- ###
        return logprob

    def get_entropy(self, probs):
        """Calculate the entropy of the action distribution.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.

        Returns:
            torch.Tensor: Entropy of the distribution, shape: (batch_size, 1)
        """
        return probs.entropy()  # Return the entropy of the probabilities

    def get_action_logprob_entropy(self, x):
        """Get action, log probability, and entropy for a given state.

        Args:
            x (torch.Tensor): Input state.

        Returns:
            tuple: (action, logprob, entropy)
                - action (torch.Tensor): Sampled action.
                - logprob (torch.Tensor): Log probability of the action.
                - entropy (torch.Tensor): Entropy of the action distribution.
        """
        probs = self.get_probs(x)  # Get the action probabilities
        action = self.get_action(probs)  # Sample an action
        logprob = self.get_action_logprob(probs, action)  # Compute log probability of the action
        entropy = self.get_entropy(probs)  # Compute entropy of the action distribution
        return action, logprob, entropy  # Return action, log probability, and entropy

    def get_deltas(self, rewards, values, next_values, next_nonterminal):
        """Compute the temporal difference (TD) error.

        Args:
            rewards (torch.Tensor): Rewards at each time step, shape: (batch_size,).
            values (torch.Tensor): Predicted values for each state, shape: (batch_size,).
            next_values (torch.Tensor): Predicted value for the next state, shape: (batch_size,).
            gamma (float): Discount factor.

        Returns:
            torch.Tensor: Computed TD errors, shape: (batch_size,).
        """
        ### -------------- TASK 2 ------------ ###
        ### ----- YOUR CODES START HERE ------ ###
        deltas = rewards + self.config.GAMMA * next_values * next_nonterminal - values
        ### ------ YOUR CODES END HERE ------- ###
        return deltas
    
    def get_ratio(self, logprob, logprob_old):
        """Compute the probability ratio between the new and old policies.

        This function calculates the ratio of the probabilities of actions under
        the current policy compared to the old policy, using their logarithmic values.

        Args:
            logprob (torch.Tensor): Log probability of the action under the current policy,
                                    shape: (batch_size,).
            logprob_old (torch.Tensor): Log probability of the action under the old policy,
                                        shape: (batch_size,).

        Returns:
            torch.Tensor: The probability ratio of the new policy to the old policy,
                        shape: (batch_size,).
        """
        ### ------------ TASK 3.1.1 ---------- ###
        ### ----- YOUR CODES START HERE ------ ###
        logratio = logprob - logprob_old  # Compute the log ratio
        ratio = torch.exp(logratio)  # Exponentiate to get the probability ratio
        ### ------ YOUR CODES END HERE ------- ###
        return ratio

    def get_policy_objective(self, advantages, ratio):
        """Compute the clipped surrogate policy objective.

        This function calculates the policy objective using the advantages and the
        probability ratio, applying clipping to stabilize training.

        Args:
            advantages (torch.Tensor): The advantage estimates, shape: (batch_size,).
            ratio (torch.Tensor): The probability ratio of the new policy to the old policy,
                                shape: (batch_size,).
            clip_coeff (float, optional): The clipping coefficient for the policy objective.
                                        Defaults to CLIP_COEF.

        Returns:
            torch.Tensor: The computed policy objective, a scalar value.
        """
        ### ------------ TASK 3.1.2 ---------- ###
        ### ----- YOUR CODES START HERE ------ ###
        policy_objective1 = ratio * advantages  # Calculate the first policy loss term
        policy_objective2 = torch.clamp(ratio, 1 - self.config.CLIP_COEF, 1 + self.config.CLIP_COEF) * advantages  # Calculate the clipped policy loss term
        policy_objective = torch.min(policy_objective1, policy_objective2).mean()  # Take the minimum and average over the batch
        ### ------ YOUR CODES END HERE ------- ###
        return policy_objective

    def get_value_loss(self, values, values_old, returns):
        """Compute the combined value loss with clipping.

        This function calculates the unclipped and clipped value losses
        and returns the maximum of the two to stabilize training.

        Args:
            values (torch.Tensor): Predicted values from the critic, shape: (batch_size, 1).
            values_old (torch.Tensor): Old predicted values from the critic, shape: (batch_size, 1).
            returns (torch.Tensor): Computed returns for the corresponding states, shape: (batch_size, 1).

        Returns:
            torch.Tensor: The combined value loss, a scalar value.
        """
        ### ------------- TASK 3.2 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        value_loss_unclipped = 0.5 * (returns - values) ** 2  # Calculate unclipped value loss

        value_loss_clipped = 0.5 * (values_old + torch.clamp(values - values_old, -self.config.CLIP_COEF, -self.config.CLIP_COEF) - returns) ** 2  # Calculate clipped value loss

        value_loss = torch.max(value_loss_clipped, value_loss_unclipped) # Maximum over the batch
        value_loss = torch.mean(value_loss)  # Mean over the batch
        ### ------ YOUR CODES END HERE ------- ###
        return value_loss  # Return the final combined value loss

    def get_entropy_objective(self, entropy):
        """Compute the entropy objective.

        This function calculates the average entropy of the action distribution,
        which encourages exploration by penalizing certainty.

        Args:
            entropy (torch.Tensor): Entropy values for the action distribution, shape: (batch_size,).

        Returns:
            torch.Tensor: The computed entropy objective, a scalar value.
        """
        return entropy.mean()  # Return the average entropy

    def get_total_loss(self, policy_objective, value_loss, entropy_objective):
        """Compute the total loss for the actor-critic agent.

        This function combines the policy objective, value loss, and entropy objective
        into a single loss value for optimization. It applies coefficients to scale
        the contribution of the value loss and entropy objective.

        Args:
            policy_objective (torch.Tensor): The policy objective, a scalar value.
            value_loss (torch.Tensor): The computed value loss, a scalar value.
            entropy_objective (torch.Tensor): The computed entropy objective, a scalar value.
            value_loss_coeff (float, optional): Coefficient for scaling the value loss. Defaults to VALUE_LOSS_COEF.
            entropy_coeff (float, optional): Coefficient for scaling the entropy loss. Defaults to ENTROPY_COEF.

        Returns:
            torch.Tensor: The total computed loss, a scalar value.
        """
        ### ------------- TASK 3.3 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        total_loss = - policy_objective + self.config.VALUE_LOSS_COEF  * value_loss - self.config.ENTROPY_COEF * entropy_objective  # Combine losses
        ### ------ YOUR CODES END HERE ------- ###
        return total_loss

    def __call__(self, observation, configuration):
        x = torch.tensor(observation["board"], dtype=torch.float32)
        x = x.unsqueeze(0)
        x = x.to(self.config.DEVICE)
        action, _, _ = self.get_action_logprob_entropy(x)
        return action.cpu().numpy()

    def train(self):
        # Initialize global step counter and reset the environment
        game_id = 0
        envs = self.env.train(self.games[game_id])
        min_reward = self.config.INVALID_MOVE_PENALTY
        max_reward = self.config.WIN_REWARD
        global_step = 0
        initial_state = envs.reset()["board"]
        state = torch.Tensor(initial_state).to(self.config.DEVICE)
        done = torch.zeros(1).to(self.config.DEVICE)

        states = torch.zeros(self.config.ROLLOUT_STEPS, self.config.input_shape).to(self.config.DEVICE)
        actions = torch.zeros(self.config.ROLLOUT_STEPS, self.config.num_actions).to(self.config.DEVICE)
        rewards = torch.zeros((self.config.ROLLOUT_STEPS, 1)).to(self.config.DEVICE)
        dones = torch.zeros((self.config.ROLLOUT_STEPS, 1)).to(self.config.DEVICE)

        logprobs = torch.zeros((self.config.ROLLOUT_STEPS, 1)).to(self.config.DEVICE)
        values = torch.zeros((self.config.ROLLOUT_STEPS, 1)).to(self.config.DEVICE)

        # Set up progress tracking
        progress_bar = tqdm(range(1, self.config.NUM_ITERATIONS + 1), postfix={'Total Rewards': 0})
        actor_loss_history = []
        critic_loss_history = []
        entropy_objective_history = []

        reward_history = []
        episode_history = []

        for iteration in progress_bar:

            # Perform rollout to gather experience
            for step in range(0, self.config.ROLLOUT_STEPS):
                global_step += 1
                states[step] = state
                dones[step] = done

                with torch.no_grad():
                    # Get action, log probability, and entropy from the agent
                    action, log_probability, _ = self.get_action_logprob_entropy(state)
                    value = self.get_value(state)
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = log_probability

                # Execute action in the environment
                next_state, reward, done, _ = envs.step(int(action.cpu().numpy()))
                mark = next_state["mark"]
                next_state = next_state["board"]
                if reward == None:
                    reward = self.config.INVALID_MOVE_PENALTY
                if done:
                    if reward == 1:
                        reward = self.config.WIN_REWARD
                    elif reward == -1:
                        reward = self.config.LOSE_PENALTY
                    elif reward == 0:
                        reward = self.config.DRAW_PENALTY
                else:
                    reward = self.get_reward(board= state, idx= self.action2index(int(action.cpu().numpy()), board= state), label= mark)
                # print(f"New reward: {reward}")
                normalized_reward = reward
                # normalized_reward = (reward - min_reward) / (max_reward - min_reward)  # Normalize the reward
                rewards[step] = torch.tensor(normalized_reward, dtype=torch.float32).to(self.config.DEVICE).view(-1)
                state = torch.Tensor(next_state).to(self.config.DEVICE)
                done = torch.Tensor([done]).to(self.config.DEVICE)

                if done:
                    reward_history.append(reward)
                    episode_history.append(global_step)
                    game_id = (game_id + 1) % len(self.games)
                    envs = self.env.train(self.games[game_id])
                    state = envs.reset()["board"]
                    state = torch.Tensor(state).to(self.config.DEVICE)

            # Calculate advantages and returns
            with torch.no_grad():
                next_value = self.get_value(state).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.config.DEVICE)

                last_gae_lambda = 0
                for t in reversed(range(self.config.ROLLOUT_STEPS)):
                    if t == self.config.ROLLOUT_STEPS - 1:
                        next_non_terminal = 1.0 - done
                        next_value = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_value = values[t + 1]

                    # Compute delta using the utility function
                    delta = self.get_deltas(rewards[t], values[t], next_value, next_non_terminal)

                    advantages[t] = last_gae_lambda = delta + self.config.GAMMA * self.config.GAE_LAMBDA * next_non_terminal * last_gae_lambda
                returns = advantages + values

            # Flatten the batch data for processing
            batch_states = states.reshape((-1,self.config.input_shape))
            batch_logprobs = logprobs.reshape(-1)
            batch_actions = actions.reshape(-1)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_values = values.reshape(-1)

            # Shuffle the batch data to break correlation between samples
            batch_indices = np.arange(self.config.BATCH_SIZE)
            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy_objective = 0

            for epoch in range(self.config.NUM_EPOCHS):
                np.random.shuffle(batch_indices)
                for start in range(0, self.config.BATCH_SIZE, self.config.MINI_BATCH_SIZE):
                    # Get the indices for the mini-batch
                    end = start + self.config.MINI_BATCH_SIZE
                    mini_batch_indices = batch_indices[start:end]

                    mini_batch_advantages = batch_advantages[mini_batch_indices]
                    # Normalize advantages to stabilize training
                    mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages.mean()) / (mini_batch_advantages.std() + 1e-8)

                    # Compute new probabilities and values for the mini-batch
                    new_probabilities = self.get_probs(batch_states[mini_batch_indices])
                    new_log_probability = self.get_action_logprob(new_probabilities, batch_actions.long()[mini_batch_indices])
                    entropy = self.get_entropy(new_probabilities)
                    new_value = self.get_value(batch_states[mini_batch_indices])

                    # Calculate the policy loss
                    ratio = self.get_ratio(new_log_probability, batch_logprobs[mini_batch_indices])
                    policy_objective = self.get_policy_objective(mini_batch_advantages, ratio)
                    policy_loss = - policy_objective

                    # Calculate the value loss
                    value_loss = self.get_value_loss(new_value.view(-1), batch_values[mini_batch_indices], batch_returns[mini_batch_indices])

                    # Calculate the entropy loss
                    entropy_objective = self.get_entropy_objective(entropy)

                    # Combine losses to get the total loss
                    total_loss = self.get_total_loss(policy_objective, value_loss, entropy_objective)

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    # Clip the gradient to stabilize training
                    nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.optimizer.step()

                    total_actor_loss += policy_loss.item()
                    total_critic_loss += value_loss.item()
                    total_entropy_objective += entropy_objective.item()

            actor_loss_history.append(total_actor_loss / self.config.NUM_EPOCHS)
            critic_loss_history.append(total_critic_loss / self.config.NUM_EPOCHS)
            entropy_objective_history.append(total_entropy_objective / self.config.NUM_EPOCHS)

            # Save model
            if (iteration + 1) % self.config.TARGET_UPDATE == 0:
                self.save()
                self.to(self.config.DEVICE)

            progress_bar.set_postfix({'Total Rewards': round(np.mean(reward_history[-100:]), 2),
                                      'Actor Loss': round(np.mean(actor_loss_history[-100:]), 2),
                                      'Critic Loss': round(np.mean(critic_loss_history[-100:]), 2),
                                      'Entropy Objective': round(np.mean(entropy_objective_history[-100:]), 2)
                                      })
    
    def actor_mul(self, total_step):

        # print("Sub-thread start")

        game_id = 0
        envs = self.env.train(self.games[game_id])
        min_reward = self.config.INVALID_MOVE_PENALTY
        max_reward = self.config.WIN_REWARD
        initial_state = envs.reset()["board"]
        state = torch.Tensor(initial_state).to(torch.device('cpu'))
        done = torch.zeros(1).to(torch.device('cpu'))

        states = torch.zeros(total_step, self.config.input_shape).to(torch.device('cpu'))
        actions = torch.zeros(total_step, self.config.num_actions).to(torch.device('cpu'))
        rewards = torch.zeros((total_step, 1)).to(torch.device('cpu'))
        dones = torch.zeros((total_step, 1)).to(torch.device('cpu'))

        logprobs = torch.zeros((total_step, 1)).to(torch.device('cpu'))
        values = torch.zeros((total_step, 1)).to(torch.device('cpu'))

        reward_history = []
        for step in range(0, total_step):
            states[step] = state
            dones[step] = done

            with torch.no_grad():
                # Get action, log probability, and entropy from the agent
                action, log_probability, _ = self.get_action_logprob_entropy(state)
                value = self.get_value(state)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = log_probability

            # Execute action in the environment
            next_state, reward, done, _ = envs.step(int(action))
            mark = next_state["mark"]
            next_state = next_state["board"]
            if reward == None:
                reward = self.config.INVALID_MOVE_PENALTY
            if done:
                if reward == 1:
                    reward = self.config.WIN_REWARD
                elif reward == -1:
                    reward = self.config.LOSE_PENALTY
                elif reward == 0:
                    reward = self.config.DRAW_PENALTY
            else:
                reward = self.get_reward(board= state, idx= self.action2index(int(action.cpu().numpy()), board= state), label= mark)
            # print(f"New reward: {reward}")
            # normalized_reward = (reward - min_reward) / (max_reward - min_reward)  # Normalize the reward
            normalized_reward = reward
            rewards[step] = torch.tensor(normalized_reward, dtype=torch.float32).to(torch.device('cpu')).view(-1)
            state = torch.Tensor(next_state).to(torch.device('cpu'))
            done = torch.Tensor([done]).to(torch.device('cpu'))

            if done:
                reward_history.append(reward)
                game_id = (game_id + 1) % len(self.games)
                envs = self.env.train(self.games[game_id])
                state = envs.reset()["board"]
                state = torch.Tensor(state).to(torch.device('cpu'))

        # Calculate advantages and returns
        with torch.no_grad():
            next_value = self.get_value(state).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(torch.device('cpu'))

            last_gae_lambda = 0
            for t in reversed(range(total_step)):
                if t == total_step - 1:
                    next_non_terminal = 1.0 - done
                    next_value = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]

                # Compute delta using the utility function
                delta = self.get_deltas(rewards[t], values[t], next_value, next_non_terminal)

                advantages[t] = last_gae_lambda = delta + self.config.GAMMA * self.config.GAE_LAMBDA * next_non_terminal * last_gae_lambda
            returns = advantages + values

        # Flatten the batch data for processing
        batch_states = states.reshape((-1,self.config.input_shape))
        batch_logprobs = logprobs.reshape(-1)
        batch_actions = actions.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)

        # print(batch_states.device, batch_actions.device, batch_logprobs.device, batch_advantages.device, batch_returns.device, batch_values.device)

        return (batch_states.cpu(), batch_actions.cpu(), batch_logprobs.cpu(), batch_advantages.cpu(), batch_returns.cpu(), batch_values.cpu(), reward_history)

    def train_mul(self, num_threads=4):
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)

        # Set up progress tracking
        actor_loss_history = []
        critic_loss_history = []
        entropy_objective_history = []

        reward_history = []

        tbar = tqdm(range(1, self.config.NUM_ITERATIONS + 1), postfix={'Total Rewards': 0})

        for iteration in tbar:

            # self.train_actor_mul(game_count=100, num_threads=num_threads)

            # All in cpu
            torch.mps.empty_cache()
            self.to(torch.device('cpu'))
            batch_states = torch.tensor([]).to(torch.device('cpu'))
            batch_actions = torch.tensor([]).to(torch.device('cpu'))
            batch_logprobs = torch.tensor([]).to(torch.device('cpu'))
            batch_advantages = torch.tensor([]).to(torch.device('cpu'))
            batch_returns = torch.tensor([]).to(torch.device('cpu'))
            batch_values = torch.tensor([]).to(torch.device('cpu'))

            # Perform rollout to gather experience
            # print("Multi-thread start")
            pool = Pool(num_threads)
            mul_result = []
            for _ in range(num_threads):
                result = pool.apply_async(self.actor_mul, args=(self.config.ROLLOUT_STEPS // num_threads, ))
                mul_result.append(result)
            pool.close()
            pool.join()

            for result in mul_result:

                # print("111")

                states, actions, logprobs, advantages, returns, values, reward = result.get()
            
                batch_states = torch.cat((batch_states, states), dim=0)
                batch_actions = torch.cat((batch_actions, actions), dim=0)
                batch_logprobs = torch.cat((batch_logprobs, logprobs), dim=0)
                batch_advantages = torch.cat((batch_advantages, advantages), dim=0)
                batch_returns = torch.cat((batch_returns, returns), dim=0)
                batch_values = torch.cat((batch_values, values), dim=0)
                reward_history.extend(reward)
            
            del mul_result, states, actions, logprobs, advantages, returns, values, reward
            del pool
            # print("Multi-thread end")

            # All in gpu
            self.to(self.config.DEVICE)
            batch_states = batch_states.to(self.config.DEVICE)
            batch_actions = batch_actions.to(self.config.DEVICE)
            batch_logprobs = batch_logprobs.to(self.config.DEVICE)
            batch_advantages = batch_advantages.to(self.config.DEVICE)
            batch_returns = batch_returns.to(self.config.DEVICE)
            batch_values = batch_values.to(self.config.DEVICE)

            # Shuffle the batch data to break correlation between samples
            batch_indices = np.arange(self.config.BATCH_SIZE)
            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy_objective = 0

            for epoch in range(self.config.NUM_EPOCHS):
                np.random.shuffle(batch_indices)
                for start in range(0, self.config.BATCH_SIZE, self.config.MINI_BATCH_SIZE):
                    # Get the indices for the mini-batch
                    end = start + self.config.MINI_BATCH_SIZE
                    mini_batch_indices = batch_indices[start:end]

                    mini_batch_advantages = batch_advantages[mini_batch_indices]
                    # Normalize advantages to stabilize training
                    mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages.mean()) / (mini_batch_advantages.std() + 1e-8)

                    # Compute new probabilities and values for the mini-batch
                    new_probabilities = self.get_probs(batch_states[mini_batch_indices])
                    new_log_probability = self.get_action_logprob(new_probabilities, batch_actions.long()[mini_batch_indices])
                    entropy = self.get_entropy(new_probabilities)
                    new_value = self.get_value(batch_states[mini_batch_indices])

                    # Calculate the policy loss
                    ratio = self.get_ratio(new_log_probability, batch_logprobs[mini_batch_indices])
                    policy_objective = self.get_policy_objective(mini_batch_advantages, ratio)
                    policy_loss = - policy_objective

                    # Calculate the value loss
                    value_loss = self.get_value_loss(new_value.view(-1), batch_values[mini_batch_indices], batch_returns[mini_batch_indices])

                    # Calculate the entropy loss
                    entropy_objective = self.get_entropy_objective(entropy)

                    # Combine losses to get the total loss
                    total_loss = self.get_total_loss(policy_objective, value_loss, entropy_objective)

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    # Clip the gradient to stabilize training
                    nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    total_actor_loss += policy_loss.item()
                    total_critic_loss += value_loss.item()
                    total_entropy_objective += entropy_objective.item()

                    del new_probabilities, new_log_probability, entropy, new_value
                    del ratio, policy_objective, policy_loss
                    del value_loss
                    del total_loss

            actor_loss_history.append(total_actor_loss / self.config.NUM_EPOCHS)
            critic_loss_history.append(total_critic_loss / self.config.NUM_EPOCHS)
            entropy_objective_history.append(total_entropy_objective / self.config.NUM_EPOCHS)

            # Save model
            if (iteration) % self.config.TARGET_UPDATE == 0:
                self.save()
                self.to(self.config.DEVICE)

            # print(f"Epoch:{iteration}, Actor Loss:{round(np.mean(actor_loss_history[-1000:]) * 100, 2)}, Critic Loss:{round(np.mean(critic_loss_history[-1000:]), 2)}, Entropy Objective:{round(np.mean(entropy_objective_history[-1000:]), 2)}, Total Rewards:{round(np.mean(reward_history[-1000:]), 2)}")
            tbar.set_postfix({'Total Rewards': round(np.mean(reward_history[-1000:]), 2),
                                      'Actor Loss': round(np.mean(actor_loss_history[-1000:]) * 100, 2),
                                      'Critic Loss': round(np.mean(critic_loss_history[-1000:]), 2),
                                      'Entropy Objective': round(np.mean(entropy_objective_history[-1000:]), 2)
                                      })

    def train_actor_mul(self, game_count= 10, num_threads=4):
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        progress_bar = tqdm(range(1, game_count + 1), postfix={'Total Rewards': 0})

        self.actor.train()
        game_id = 0
        for iteration in progress_bar:
            # All in cpu
            torch.mps.empty_cache()
            self.to(torch.device('cpu'))
            batch_states = torch.tensor([]).to(torch.device('cpu'))
            batch_actions = torch.tensor([]).to(torch.device('cpu'))
            batch_logprobs = torch.tensor([]).to(torch.device('cpu'))
            batch_advantages = torch.tensor([]).to(torch.device('cpu'))
            batch_returns = torch.tensor([]).to(torch.device('cpu'))
            batch_values = torch.tensor([]).to(torch.device('cpu'))

            # Perform rollout to gather experience
            # print("Multi-thread start")
            pool = Pool(num_threads)
            mul_result = []
            for _ in range(num_threads):
                result = pool.apply_async(self.actor_mul, args=(self.config.ROLLOUT_STEPS // num_threads, ))
                mul_result.append(result)
            pool.close()
            pool.join()

            for result in mul_result:

                # print("111")

                states, actions, logprobs, advantages, returns, values, reward = result.get()
            
                batch_states = torch.cat((batch_states, states), dim=0)
                batch_actions = torch.cat((batch_actions, actions), dim=0)
                batch_logprobs = torch.cat((batch_logprobs, logprobs), dim=0)
                batch_advantages = torch.cat((batch_advantages, advantages), dim=0)
                batch_returns = torch.cat((batch_returns, returns), dim=0)
                batch_values = torch.cat((batch_values, values), dim=0)
                self.rewards.extend(reward)
            
            del mul_result, states, actions, logprobs, advantages, returns, values, reward
            del pool
            # print("Multi-thread end")

            batch_indices = np.arange(self.config.BATCH_SIZE)

            total_actor_loss = 0

            for epoch in range(self.config.NUM_EPOCHS):
                np.random.shuffle(batch_indices)
                for start in range(0, self.config.BATCH_SIZE, self.config.MINI_BATCH_SIZE):
                    # Get the indices for the mini-batch
                    end = start + self.config.MINI_BATCH_SIZE
                    mini_batch_indices = batch_indices[start:end]

                    mini_batch_advantages = batch_advantages[mini_batch_indices]
                    # Normalize advantages to stabilize training
                    mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages.mean()) / (mini_batch_advantages.std() + 1e-8)

                    # Compute new probabilities and values for the mini-batch
                    new_probabilities = self.get_probs(batch_states[mini_batch_indices])
                    new_log_probability = self.get_action_logprob(new_probabilities, batch_actions.long()[mini_batch_indices])

                    # Calculate the policy loss
                    ratio = self.get_ratio(new_log_probability, batch_logprobs[mini_batch_indices])
                    policy_objective = self.get_policy_objective(mini_batch_advantages, ratio)
                    policy_loss = - policy_objective

                    self.optimizer.zero_grad()
                    policy_loss.backward()
                    # Clip the gradient to stabilize training
                    nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.optimizer.step()

                    total_actor_loss += policy_loss.item()

                    del new_probabilities, new_log_probability
                    del ratio, policy_objective, policy_loss

            self.loss.append(total_actor_loss / self.config.NUM_EPOCHS)

            progress_bar.set_postfix({'Total Rewards': round(np.mean(self.rewards[-1000:]), 2),
                                      'Actor Loss': round(np.mean(self.loss[-1000:]) * 100, 2)
                                      })

            # Save model
            if (iteration + 1) % self.config.TARGET_UPDATE == 0:
                self.save()
                self.to(self.config.DEVICE)

class PPO_CNN_agent(Agent):
    def __init__(self, model_name= "PPO"):
        """Initialize the Actor-Critic agent with actor and critic networks."""
        super().__init__(model_name)

        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_objective_history = []

        self.reward_history = []
        self.episode_history = []

        ### ------------- TASK 1.1 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        actor_input_dim = self.config.input_shape  # Input dimension for the actor
        actor_output_dim = self.config.num_actions  # Output dimension for the actor (number of actions)
        critic_input_dim = self.config.input_shape  # Input dimension for the critic
        critic_output_dim = 1  # Output dimension for the critic (value estimate)
        ### ------ YOUR CODES END HERE ------- ###

        # Define the actor network
        self.actor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 6, 64)),
            nn.Tanh(), 
            layer_init(nn.Linear(64, actor_output_dim), std=0.01),  # Final layer with small std for output
        )

        # Define the critic network
        self.critic = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), 
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 6, 64)),
            nn.Tanh(), 
            layer_init(nn.Linear(64, critic_output_dim), std=1.0),  # Standard output layer for value
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)

        self.to(self.config.DEVICE)

    def get_value(self, x):
        """Calculate the estimated value for a given state.

        Args:
            x (torch.Tensor): Input state, shape: (batch_size, observation_size)

        Returns:
            torch.Tensor: Estimated value for the state, shape: (batch_size, 1)
        """
        ### ------------- TASK 1.2 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        x = x.view(-1, 1, 6, 7)
        value = self.critic(x)  # Forward pass through the critic network
        ### ------ YOUR CODES END HERE ------- ###
        return value

    def get_probs(self, x):
        """Calculate the action probabilities for a given state.

        Args:
            x (torch.Tensor): Input state, shape: (batch_size, observation_size)

        Returns:
            torch.distributions.Categorical: Categorical distribution over actions.
        """
        ### ------------- TASK 1.3 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        x = x.view(-1, 1, 6, 7)
        logits = self.actor(x)  # Get logits from the actor network
        probs = torch.distributions.Categorical(logits=logits)  # Create a categorical distribution from the logits
        ### ------ YOUR CODES END HERE ------- ###
        return probs

    def get_action(self, probs):
        """Sample an action from the action probabilities.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.

        Returns:
            torch.Tensor: Sampled action, shape: (batch_size, 1)
        """
        ### ------------- TASK 1.4 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        action = probs.sample()  # Sample an action based on the probabilities
        ### ------ YOUR CODES END HERE ------- ###
        return action

    def get_action_logprob(self, probs, action):
        """Compute the log probability of a given action.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.
            action (torch.Tensor): Selected action, shape: (batch_size, 1)

        Returns:
            torch.Tensor: Log probability of the action, shape: (batch_size, 1)
        """
        ### ------------- TASK 1.5 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        logprob = probs.log_prob(action)  # Calculate log probability of the sampled action
        ### ------ YOUR CODES END HERE ------- ###
        return logprob

    def get_entropy(self, probs):
        """Calculate the entropy of the action distribution.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.

        Returns:
            torch.Tensor: Entropy of the distribution, shape: (batch_size, 1)
        """
        return probs.entropy()  # Return the entropy of the probabilities

    def get_action_logprob_entropy(self, x):
        """Get action, log probability, and entropy for a given state.

        Args:
            x (torch.Tensor): Input state.

        Returns:
            tuple: (action, logprob, entropy)
                - action (torch.Tensor): Sampled action.
                - logprob (torch.Tensor): Log probability of the action.
                - entropy (torch.Tensor): Entropy of the action distribution.
        """
        probs = self.get_probs(x)  # Get the action probabilities
        action = self.get_action(probs)  # Sample an action
        logprob = self.get_action_logprob(probs, action)  # Compute log probability of the action
        entropy = self.get_entropy(probs)  # Compute entropy of the action distribution
        return action, logprob, entropy  # Return action, log probability, and entropy

    def get_deltas(self, rewards, values, next_values, next_nonterminal):
        """Compute the temporal difference (TD) error.

        Args:
            rewards (torch.Tensor): Rewards at each time step, shape: (batch_size,).
            values (torch.Tensor): Predicted values for each state, shape: (batch_size,).
            next_values (torch.Tensor): Predicted value for the next state, shape: (batch_size,).
            gamma (float): Discount factor.

        Returns:
            torch.Tensor: Computed TD errors, shape: (batch_size,).
        """
        ### -------------- TASK 2 ------------ ###
        ### ----- YOUR CODES START HERE ------ ###
        deltas = rewards + self.config.GAMMA * next_values * next_nonterminal - values
        ### ------ YOUR CODES END HERE ------- ###
        return deltas
    
    def get_ratio(self, logprob, logprob_old):
        """Compute the probability ratio between the new and old policies.

        This function calculates the ratio of the probabilities of actions under
        the current policy compared to the old policy, using their logarithmic values.

        Args:
            logprob (torch.Tensor): Log probability of the action under the current policy,
                                    shape: (batch_size,).
            logprob_old (torch.Tensor): Log probability of the action under the old policy,
                                        shape: (batch_size,).

        Returns:
            torch.Tensor: The probability ratio of the new policy to the old policy,
                        shape: (batch_size,).
        """
        ### ------------ TASK 3.1.1 ---------- ###
        ### ----- YOUR CODES START HERE ------ ###
        logratio = logprob - logprob_old  # Compute the log ratio
        ratio = torch.exp(logratio)  # Exponentiate to get the probability ratio
        ### ------ YOUR CODES END HERE ------- ###
        return ratio

    def get_policy_objective(self, advantages, ratio):
        """Compute the clipped surrogate policy objective.

        This function calculates the policy objective using the advantages and the
        probability ratio, applying clipping to stabilize training.

        Args:
            advantages (torch.Tensor): The advantage estimates, shape: (batch_size,).
            ratio (torch.Tensor): The probability ratio of the new policy to the old policy,
                                shape: (batch_size,).
            clip_coeff (float, optional): The clipping coefficient for the policy objective.
                                        Defaults to CLIP_COEF.

        Returns:
            torch.Tensor: The computed policy objective, a scalar value.
        """
        ### ------------ TASK 3.1.2 ---------- ###
        ### ----- YOUR CODES START HERE ------ ###
        policy_objective1 = ratio * advantages  # Calculate the first policy loss term
        policy_objective2 = torch.clamp(ratio, 1 - self.config.CLIP_COEF, 1 + self.config.CLIP_COEF) * advantages  # Calculate the clipped policy loss term
        policy_objective = torch.min(policy_objective1, policy_objective2).mean()  # Take the minimum and average over the batch
        ### ------ YOUR CODES END HERE ------- ###
        return policy_objective

    def get_value_loss(self, values, values_old, returns):
        """Compute the combined value loss with clipping.

        This function calculates the unclipped and clipped value losses
        and returns the maximum of the two to stabilize training.

        Args:
            values (torch.Tensor): Predicted values from the critic, shape: (batch_size, 1).
            values_old (torch.Tensor): Old predicted values from the critic, shape: (batch_size, 1).
            returns (torch.Tensor): Computed returns for the corresponding states, shape: (batch_size, 1).

        Returns:
            torch.Tensor: The combined value loss, a scalar value.
        """
        ### ------------- TASK 3.2 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        value_loss_unclipped = 0.5 * (returns - values) ** 2  # Calculate unclipped value loss

        value_loss_clipped = 0.5 * (values_old + torch.clamp(values - values_old, -self.config.CLIP_COEF, -self.config.CLIP_COEF) - returns) ** 2  # Calculate clipped value loss

        value_loss = torch.max(value_loss_clipped, value_loss_unclipped) # Maximum over the batch
        value_loss = torch.mean(value_loss)  # Mean over the batch
        ### ------ YOUR CODES END HERE ------- ###
        return value_loss  # Return the final combined value loss

    def get_entropy_objective(self, entropy):
        """Compute the entropy objective.

        This function calculates the average entropy of the action distribution,
        which encourages exploration by penalizing certainty.

        Args:
            entropy (torch.Tensor): Entropy values for the action distribution, shape: (batch_size,).

        Returns:
            torch.Tensor: The computed entropy objective, a scalar value.
        """
        return entropy.mean()  # Return the average entropy

    def get_total_loss(self, policy_objective, value_loss, entropy_objective):
        """Compute the total loss for the actor-critic agent.

        This function combines the policy objective, value loss, and entropy objective
        into a single loss value for optimization. It applies coefficients to scale
        the contribution of the value loss and entropy objective.

        Args:
            policy_objective (torch.Tensor): The policy objective, a scalar value.
            value_loss (torch.Tensor): The computed value loss, a scalar value.
            entropy_objective (torch.Tensor): The computed entropy objective, a scalar value.
            value_loss_coeff (float, optional): Coefficient for scaling the value loss. Defaults to VALUE_LOSS_COEF.
            entropy_coeff (float, optional): Coefficient for scaling the entropy loss. Defaults to ENTROPY_COEF.

        Returns:
            torch.Tensor: The total computed loss, a scalar value.
        """
        ### ------------- TASK 3.3 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        total_loss = - policy_objective + self.config.VALUE_LOSS_COEF  * value_loss - self.config.ENTROPY_COEF * entropy_objective  # Combine losses
        ### ------ YOUR CODES END HERE ------- ###
        return total_loss

    def __call__(self, observation, configuration):
        x = torch.tensor(observation["board"], dtype=torch.float32)
        x = x.unsqueeze(0)
        x = x.to(self.config.DEVICE)
        action, _, _ = self.get_action_logprob_entropy(x)
        return action.cpu().numpy()

    def train(self):
        # Initialize global step counter and reset the environment
        game_id = 0
        envs = self.env.train(self.games[game_id])
        min_reward = self.config.INVALID_MOVE_PENALTY
        max_reward = self.config.WIN_REWARD
        global_step = 0
        initial_state = envs.reset()["board"]
        state = torch.Tensor(initial_state).to(self.config.DEVICE)
        done = torch.zeros(1).to(self.config.DEVICE)

        states = torch.zeros(self.config.ROLLOUT_STEPS, self.config.input_shape).to(self.config.DEVICE)
        actions = torch.zeros(self.config.ROLLOUT_STEPS, self.config.num_actions).to(self.config.DEVICE)
        rewards = torch.zeros((self.config.ROLLOUT_STEPS, 1)).to(self.config.DEVICE)
        dones = torch.zeros((self.config.ROLLOUT_STEPS, 1)).to(self.config.DEVICE)

        logprobs = torch.zeros((self.config.ROLLOUT_STEPS, 1)).to(self.config.DEVICE)
        values = torch.zeros((self.config.ROLLOUT_STEPS, 1)).to(self.config.DEVICE)

        # Set up progress tracking
        progress_bar = tqdm(range(1, self.config.NUM_ITERATIONS + 1), postfix={'Total Rewards': 0})
        actor_loss_history = []
        critic_loss_history = []
        entropy_objective_history = []

        reward_history = []
        episode_history = []

        for iteration in progress_bar:

            # Perform rollout to gather experience
            for step in range(0, self.config.ROLLOUT_STEPS):
                global_step += 1
                states[step] = state
                dones[step] = done

                with torch.no_grad():
                    # Get action, log probability, and entropy from the agent
                    action, log_probability, _ = self.get_action_logprob_entropy(state)
                    value = self.get_value(state)
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = log_probability

                # Execute action in the environment
                next_state, reward, done, _ = envs.step(int(action.cpu().numpy()))
                mark = next_state["mark"]
                next_state = next_state["board"]
                if reward == None:
                    reward = self.config.INVALID_MOVE_PENALTY
                if done:
                    if reward == 1:
                        reward = self.config.WIN_REWARD
                    elif reward == -1:
                        reward = self.config.LOSE_PENALTY
                    elif reward == 0:
                        reward = self.config.DRAW_PENALTY
                else:
                    reward = self.get_reward(board= state, idx= self.action2index(int(action.cpu().numpy()), board= state), label= mark)
                # print(f"New reward: {reward}")
                normalized_reward = reward
                # normalized_reward = (reward - min_reward) / (max_reward - min_reward)  # Normalize the reward
                rewards[step] = torch.tensor(normalized_reward, dtype=torch.float32).to(self.config.DEVICE).view(-1)
                state = torch.Tensor(next_state).to(self.config.DEVICE)
                done = torch.Tensor([done]).to(self.config.DEVICE)

                if done:
                    reward_history.append(reward)
                    episode_history.append(global_step)
                    game_id = (game_id + 1) % len(self.games)
                    envs = self.env.train(self.games[game_id])
                    state = envs.reset()["board"]
                    state = torch.Tensor(state).to(self.config.DEVICE)

            # Calculate advantages and returns
            with torch.no_grad():
                next_value = self.get_value(state).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.config.DEVICE)

                last_gae_lambda = 0
                for t in reversed(range(self.config.ROLLOUT_STEPS)):
                    if t == self.config.ROLLOUT_STEPS - 1:
                        next_non_terminal = 1.0 - done
                        next_value = next_value
                    else:
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_value = values[t + 1]

                    # Compute delta using the utility function
                    delta = self.get_deltas(rewards[t], values[t], next_value, next_non_terminal)

                    advantages[t] = last_gae_lambda = delta + self.config.GAMMA * self.config.GAE_LAMBDA * next_non_terminal * last_gae_lambda
                returns = advantages + values

            # Flatten the batch data for processing
            batch_states = states.reshape((-1,self.config.input_shape))
            batch_logprobs = logprobs.reshape(-1)
            batch_actions = actions.reshape(-1)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_values = values.reshape(-1)

            # Shuffle the batch data to break correlation between samples
            batch_indices = np.arange(self.config.BATCH_SIZE)
            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy_objective = 0

            for epoch in range(self.config.NUM_EPOCHS):
                np.random.shuffle(batch_indices)
                for start in range(0, self.config.BATCH_SIZE, self.config.MINI_BATCH_SIZE):
                    # Get the indices for the mini-batch
                    end = start + self.config.MINI_BATCH_SIZE
                    mini_batch_indices = batch_indices[start:end]

                    mini_batch_advantages = batch_advantages[mini_batch_indices]
                    # Normalize advantages to stabilize training
                    mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages.mean()) / (mini_batch_advantages.std() + 1e-8)

                    # Compute new probabilities and values for the mini-batch
                    new_probabilities = self.get_probs(batch_states[mini_batch_indices])
                    new_log_probability = self.get_action_logprob(new_probabilities, batch_actions.long()[mini_batch_indices])
                    entropy = self.get_entropy(new_probabilities)
                    new_value = self.get_value(batch_states[mini_batch_indices])

                    # Calculate the policy loss
                    ratio = self.get_ratio(new_log_probability, batch_logprobs[mini_batch_indices])
                    policy_objective = self.get_policy_objective(mini_batch_advantages, ratio)
                    policy_loss = - policy_objective

                    # Calculate the value loss
                    value_loss = self.get_value_loss(new_value.view(-1), batch_values[mini_batch_indices], batch_returns[mini_batch_indices])

                    # Calculate the entropy loss
                    entropy_objective = self.get_entropy_objective(entropy)

                    # Combine losses to get the total loss
                    total_loss = self.get_total_loss(policy_objective, value_loss, entropy_objective)

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    # Clip the gradient to stabilize training
                    nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.optimizer.step()

                    total_actor_loss += policy_loss.item()
                    total_critic_loss += value_loss.item()
                    total_entropy_objective += entropy_objective.item()

            actor_loss_history.append(total_actor_loss / self.config.NUM_EPOCHS)
            critic_loss_history.append(total_critic_loss / self.config.NUM_EPOCHS)
            entropy_objective_history.append(total_entropy_objective / self.config.NUM_EPOCHS)

            # Save model
            if (iteration + 1) % self.config.TARGET_UPDATE == 0:
                self.save()
                self.to(self.config.DEVICE)

            progress_bar.set_postfix({'Total Rewards': round(np.mean(reward_history[-100:]), 2),
                                      'Actor Loss': round(np.mean(actor_loss_history[-100:]), 2),
                                      'Critic Loss': round(np.mean(critic_loss_history[-100:]), 2),
                                      'Entropy Objective': round(np.mean(entropy_objective_history[-100:]), 2)
                                      })
    
    def actor_mul(self, total_step):

        # print("Sub-thread start")

        game_id = 0
        envs = self.env.train(self.games[game_id])
        min_reward = self.config.INVALID_MOVE_PENALTY
        max_reward = self.config.WIN_REWARD
        initial_state = envs.reset()["board"]
        state = torch.Tensor(initial_state).to(torch.device('cpu'))
        done = torch.zeros(1).to(torch.device('cpu'))

        states = torch.zeros(total_step, self.config.input_shape).to(torch.device('cpu'))
        actions = torch.zeros(total_step, self.config.num_actions).to(torch.device('cpu'))
        rewards = torch.zeros((total_step, 1)).to(torch.device('cpu'))
        dones = torch.zeros((total_step, 1)).to(torch.device('cpu'))

        logprobs = torch.zeros((total_step, 1)).to(torch.device('cpu'))
        values = torch.zeros((total_step, 1)).to(torch.device('cpu'))

        reward_history = []
        for step in range(0, total_step):
            states[step] = state
            dones[step] = done

            with torch.no_grad():
                # Get action, log probability, and entropy from the agent
                action, log_probability, _ = self.get_action_logprob_entropy(state)
                value = self.get_value(state)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = log_probability

            # Execute action in the environment
            next_state, reward, done, _ = envs.step(int(action))
            mark = next_state["mark"]
            next_state = next_state["board"]
            if reward == None:
                reward = self.config.INVALID_MOVE_PENALTY
            if done:
                if reward == 1:
                    reward = self.config.WIN_REWARD
                elif reward == -1:
                    reward = self.config.LOSE_PENALTY
                elif reward == 0:
                    reward = self.config.DRAW_PENALTY
            else:
                reward = self.get_reward(board= state, idx= self.action2index(int(action.cpu().numpy()), board= state), label= mark)
            # print(f"New reward: {reward}")
            # normalized_reward = (reward - min_reward) / (max_reward - min_reward)  # Normalize the reward
            normalized_reward = reward
            rewards[step] = torch.tensor(normalized_reward, dtype=torch.float32).to(torch.device('cpu')).view(-1)
            state = torch.Tensor(next_state).to(torch.device('cpu'))
            done = torch.Tensor([done]).to(torch.device('cpu'))

            if done:
                reward_history.append(reward)
                game_id = (game_id + 1) % len(self.games)
                envs = self.env.train(self.games[game_id])
                state = envs.reset()["board"]
                state = torch.Tensor(state).to(torch.device('cpu'))

        # Calculate advantages and returns
        with torch.no_grad():
            next_value = self.get_value(state).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(torch.device('cpu'))

            last_gae_lambda = 0
            for t in reversed(range(total_step)):
                if t == total_step - 1:
                    next_non_terminal = 1.0 - done
                    next_value = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]

                # Compute delta using the utility function
                delta = self.get_deltas(rewards[t], values[t], next_value, next_non_terminal)

                advantages[t] = last_gae_lambda = delta + self.config.GAMMA * self.config.GAE_LAMBDA * next_non_terminal * last_gae_lambda
            returns = advantages + values

        # Flatten the batch data for processing
        batch_states = states.reshape((-1,self.config.input_shape))
        batch_logprobs = logprobs.reshape(-1)
        batch_actions = actions.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)

        # print(batch_states.device, batch_actions.device, batch_logprobs.device, batch_advantages.device, batch_returns.device, batch_values.device)

        return (batch_states.cpu(), batch_actions.cpu(), batch_logprobs.cpu(), batch_advantages.cpu(), batch_returns.cpu(), batch_values.cpu(), reward_history)

    def train_mul(self, num_threads=4):
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)

        # Set up progress tracking
        actor_loss_history = []
        critic_loss_history = []
        entropy_objective_history = []

        reward_history = []

        for iteration in range(1, self.config.NUM_ITERATIONS + 1):

            # self.train_actor_mul(game_count=100, num_threads=num_threads)

            # All in cpu
            torch.mps.empty_cache()
            self.to(torch.device('cpu'))
            batch_states = torch.tensor([]).to(torch.device('cpu'))
            batch_actions = torch.tensor([]).to(torch.device('cpu'))
            batch_logprobs = torch.tensor([]).to(torch.device('cpu'))
            batch_advantages = torch.tensor([]).to(torch.device('cpu'))
            batch_returns = torch.tensor([]).to(torch.device('cpu'))
            batch_values = torch.tensor([]).to(torch.device('cpu'))

            # Perform rollout to gather experience
            # print("Multi-thread start")
            pool = Pool(num_threads)
            mul_result = []
            for _ in range(num_threads):
                result = pool.apply_async(self.actor_mul, args=(self.config.ROLLOUT_STEPS // num_threads, ))
                mul_result.append(result)
            pool.close()
            pool.join()

            for result in mul_result:

                # print("111")

                states, actions, logprobs, advantages, returns, values, reward = result.get()
            
                batch_states = torch.cat((batch_states, states), dim=0)
                batch_actions = torch.cat((batch_actions, actions), dim=0)
                batch_logprobs = torch.cat((batch_logprobs, logprobs), dim=0)
                batch_advantages = torch.cat((batch_advantages, advantages), dim=0)
                batch_returns = torch.cat((batch_returns, returns), dim=0)
                batch_values = torch.cat((batch_values, values), dim=0)
                reward_history.extend(reward)
            
            del mul_result, states, actions, logprobs, advantages, returns, values, reward
            del pool
            # print("Multi-thread end")

            # All in gpu
            self.to(self.config.DEVICE)
            batch_states = batch_states.to(self.config.DEVICE)
            batch_actions = batch_actions.to(self.config.DEVICE)
            batch_logprobs = batch_logprobs.to(self.config.DEVICE)
            batch_advantages = batch_advantages.to(self.config.DEVICE)
            batch_returns = batch_returns.to(self.config.DEVICE)
            batch_values = batch_values.to(self.config.DEVICE)

            # Shuffle the batch data to break correlation between samples
            batch_indices = np.arange(self.config.BATCH_SIZE)
            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy_objective = 0

            for epoch in range(self.config.NUM_EPOCHS):
                np.random.shuffle(batch_indices)
                for start in range(0, self.config.BATCH_SIZE, self.config.MINI_BATCH_SIZE):
                    # Get the indices for the mini-batch
                    end = start + self.config.MINI_BATCH_SIZE
                    mini_batch_indices = batch_indices[start:end]

                    mini_batch_advantages = batch_advantages[mini_batch_indices]
                    # Normalize advantages to stabilize training
                    mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages.mean()) / (mini_batch_advantages.std() + 1e-8)

                    # Compute new probabilities and values for the mini-batch
                    new_probabilities = self.get_probs(batch_states[mini_batch_indices])
                    new_log_probability = self.get_action_logprob(new_probabilities, batch_actions.long()[mini_batch_indices])
                    entropy = self.get_entropy(new_probabilities)
                    new_value = self.get_value(batch_states[mini_batch_indices])

                    # Calculate the policy loss
                    ratio = self.get_ratio(new_log_probability, batch_logprobs[mini_batch_indices])
                    policy_objective = self.get_policy_objective(mini_batch_advantages, ratio)
                    policy_loss = - policy_objective

                    # Calculate the value loss
                    value_loss = self.get_value_loss(new_value.view(-1), batch_values[mini_batch_indices], batch_returns[mini_batch_indices])

                    # Calculate the entropy loss
                    entropy_objective = self.get_entropy_objective(entropy)

                    # Combine losses to get the total loss
                    total_loss = self.get_total_loss(policy_objective, value_loss, entropy_objective)

                    self.optimizer.zero_grad()
                    policy_loss.backward()
                    # Clip the gradient to stabilize training
                    nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    total_actor_loss += policy_loss.item()
                    total_critic_loss += value_loss.item()
                    total_entropy_objective += entropy_objective.item()

                    del new_probabilities, new_log_probability, entropy, new_value
                    del ratio, policy_objective, policy_loss
                    del value_loss
                    del total_loss

            actor_loss_history.append(total_actor_loss / self.config.NUM_EPOCHS)
            critic_loss_history.append(total_critic_loss / self.config.NUM_EPOCHS)
            entropy_objective_history.append(total_entropy_objective / self.config.NUM_EPOCHS)

            # Save model
            if (iteration) % self.config.TARGET_UPDATE == 0:
                self.save()
                self.to(self.config.DEVICE)

            print(f"Epoch:{iteration}, Actor Loss:{round(np.mean(actor_loss_history[-1000:]) * 100, 2)}, Critic Loss:{round(np.mean(critic_loss_history[-1000:]), 2)}, Entropy Objective:{round(np.mean(entropy_objective_history[-1000:]), 2)}, Total Rewards:{round(np.mean(reward_history[-1000:]), 2)}")

    def train_actor_mul(self, game_count= 10, num_threads=4):
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        progress_bar = tqdm(range(1, game_count + 1), postfix={'Total Rewards': 0})

        self.actor.train()
        game_id = 0
        for iteration in progress_bar:
            # All in cpu
            torch.mps.empty_cache()
            self.to(torch.device('cpu'))
            batch_states = torch.tensor([]).to(torch.device('cpu'))
            batch_actions = torch.tensor([]).to(torch.device('cpu'))
            batch_logprobs = torch.tensor([]).to(torch.device('cpu'))
            batch_advantages = torch.tensor([]).to(torch.device('cpu'))
            batch_returns = torch.tensor([]).to(torch.device('cpu'))
            batch_values = torch.tensor([]).to(torch.device('cpu'))

            # Perform rollout to gather experience
            # print("Multi-thread start")
            pool = Pool(num_threads)
            mul_result = []
            for _ in range(num_threads):
                result = pool.apply_async(self.actor_mul, args=(self.config.ROLLOUT_STEPS // num_threads, ))
                mul_result.append(result)
            pool.close()
            pool.join()

            for result in mul_result:

                # print("111")

                states, actions, logprobs, advantages, returns, values, reward = result.get()
            
                batch_states = torch.cat((batch_states, states), dim=0)
                batch_actions = torch.cat((batch_actions, actions), dim=0)
                batch_logprobs = torch.cat((batch_logprobs, logprobs), dim=0)
                batch_advantages = torch.cat((batch_advantages, advantages), dim=0)
                batch_returns = torch.cat((batch_returns, returns), dim=0)
                batch_values = torch.cat((batch_values, values), dim=0)
                self.rewards.extend(reward)
            
            del mul_result, states, actions, logprobs, advantages, returns, values, reward
            del pool
            # print("Multi-thread end")

            batch_indices = np.arange(self.config.BATCH_SIZE)

            total_actor_loss = 0

            for epoch in range(self.config.NUM_EPOCHS):
                np.random.shuffle(batch_indices)
                for start in range(0, self.config.BATCH_SIZE, self.config.MINI_BATCH_SIZE):
                    # Get the indices for the mini-batch
                    end = start + self.config.MINI_BATCH_SIZE
                    mini_batch_indices = batch_indices[start:end]

                    mini_batch_advantages = batch_advantages[mini_batch_indices]
                    # Normalize advantages to stabilize training
                    mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages.mean()) / (mini_batch_advantages.std() + 1e-8)

                    # Compute new probabilities and values for the mini-batch
                    new_probabilities = self.get_probs(batch_states[mini_batch_indices])
                    new_log_probability = self.get_action_logprob(new_probabilities, batch_actions.long()[mini_batch_indices])

                    # Calculate the policy loss
                    ratio = self.get_ratio(new_log_probability, batch_logprobs[mini_batch_indices])
                    policy_objective = self.get_policy_objective(mini_batch_advantages, ratio)
                    policy_loss = - policy_objective

                    self.optimizer.zero_grad()
                    policy_loss.backward()
                    # Clip the gradient to stabilize training
                    nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.optimizer.step()

                    total_actor_loss += policy_loss.item()

                    del new_probabilities, new_log_probability
                    del ratio, policy_objective, policy_loss

            self.loss.append(total_actor_loss / self.config.NUM_EPOCHS)

            progress_bar.set_postfix({'Total Rewards': round(np.mean(self.rewards[-1000:]), 2),
                                      'Actor Loss': round(np.mean(self.loss[-1000:]) * 100, 2)
                                      })

            # Save model
            if (iteration + 1) % self.config.TARGET_UPDATE == 0:
                self.save()
                self.to(self.config.DEVICE)



class PolicyAgent(nn.Module):
    def __init__(self,input_dim=42,output_dim=7,embed_dim=256):
        super(PolicyAgent,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.embed_dim=embed_dim
        self.head=nn.Sequential(
            nn.Linear(self.input_dim,self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            
            nn.ReLU(),
            nn.Linear(self.embed_dim,2*self.embed_dim),
            nn.BatchNorm1d(2*self.embed_dim),
            
            nn.ReLU(),
            nn.Linear(2*self.embed_dim,self.output_dim),
        )
        
    def forward(self,x):
        x=self.head(x)
        return F.softmax(x,dim=-1)

#用于评估当前状态下的价值
class ValueAgent(nn.Module):
    def __init__(self,input_dim=42,output_dim=1,embed_dim=256):
        super(ValueAgent,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.embed_dim=embed_dim
        self.head=nn.Sequential(
            nn.Linear(self.input_dim,self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            
            nn.ReLU(),
            nn.Linear(self.embed_dim,2*self.embed_dim),
            nn.BatchNorm1d(2*self.embed_dim),
            
            nn.ReLU(),
            nn.Linear(2*self.embed_dim,self.output_dim),
        )
    def loss_fn(self,y_true,y_pred):
        return torch.mean((y_true-y_pred)**2)
        
    def forward(self,x):
        x = x.reshape(x.size(0), -1)
        print(x.shape)
        x=self.head(x)
        return x

class PPO_Agent_v2(Agent):
    def __init__(self, model_name= "PPO"):
        """Initialize the Actor-Critic agent with actor and critic networks."""
        super().__init__(model_name)
        self.policyagent = PolicyAgent(input_dim= self.config.input_shape,output_dim= self.config.num_actions,embed_dim= 256)
        self.valueagent = ValueAgent(input_dim= self.config.input_shape,output_dim= 1,embed_dim= 256)
        self.policyoptimizer = optim.AdamW(self.policyagent.parameters(), lr=self.config.LEARNING_RATE, betas=(0.5,0.999))
        self.valueoptimizer = optim.AdamW(self.valueagent.parameters(), lr=self.config.LEARNING_RATE, betas=(0.5,0.999))

        self.win_reward = self.config.win_reward#赢了这场比赛能够得到的回报
        self.draw_reward = self.config.draw_reward#如果比赛平局能够得到的回报
        self.loss_reward = self.config.loss_reward#输掉一场比赛能够得到的reward
        self.error_reward = self.config.error_reward#走到一个不能走的位置,那要严重警告
        self.discount = self.config.discount#折扣率
        self.lmbda = self.config.GAE_LAMBDA#gae的超参数
        self.clip_ratio = self.config.CLIP_COEF#clipping系数

        self.to(self.config.DEVICE)
    
    def __call__(self, observation, configuration):
        state = observation["board"]
        idx=torch.where(torch.sum(state.reshape(6,7)==0,axis=0)==0)[0]#某列上=0的个数为0,棋子摆满了
        self.policyagent.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.config.DEVICE)
            action_pros=self.policyagent(state_tensor).cpu().detach().numpy()
            if len(idx):
                action_pros[0,idx]=0#有棋子的地方概率为0
            action=np.argmax(action_pros)
        return int(action)

    def train(self,game_counts=400001,every=1000,epochs=5):
        #game_counts:总共进行多少场游戏,every:每多少场游戏检查一下输赢及error情况
        #run_steps:每走几步训练一次模型,epochs模型训练几个epoch
        
        total_rewards=[]#这里统计每次训练的结果
        states=[]#每个状态保存
        actions=[]#每个状态后采取的action
        rewards=[]#采取action后得到的reward
        old_policys=[]#每次action的概率分布
        one_game_rewards=[]#一场游戏的rewards
        
        for game_count in range(game_counts):#开始玩第几场游戏
            
            done=False#游戏目前还没有结束
            train_env = self.env.train(random.choice(self.games)) #初始化环境
            
            state = [train_env.reset()['board']] #环境进行初始化
            while not done:#游戏还没有结束就继续玩:
                self.policyagent.eval()#策略网络先换成评估模式
                #采取每个action的概率
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.config.DEVICE)
                action_probs=self.policyagent(state_tensor).cpu().detach().numpy()[0]
                #从action中轮盘赌随机选择一个action
                action=np.random.choice(len(action_probs),p=action_probs)
                #我执行了action之后得到了下一个状态next_state,奖励reward,done游戏是否结束
                next_state, reward, done, _ = train_env.step(action)
                #next_state转成(42)的np.array
                next_state=[next_state['board']]#更新下一步
                
                #对reward的调整
                if done:#如果一场游戏结束
                    if reward is None:#如果reward是空的
                        reward = self.error_reward#说明游戏结束时发生了异常情况
                        done = True #游戏结束了
                    elif reward == 1:#赢了就给赢了的reward
                        reward = self.win_reward
                    elif reward == -1:#输了就给输了的reward
                        reward = self.loss_reward
                    elif reward == 0:#平局就给平局的reward
                        reward = self.draw_reward
                else:#如果没有结束游戏,就给智能体一点小惩罚,鼓励智能体快点结束游戏。
                    reward = -1/30
                
                
                states.append(state)#当前的state
                actions.append(action)#当前state随便选一个action
                rewards.append(reward)#当前的state走action后得到的reward
                old_policys.append(action_probs)#7个action的概率
                
                #需要训练模型了,run_steps就是游戏走了几步,搜集了一批数据,done就是游戏玩完了
                if (len(states)>self.config.BATCH_SIZE) or (done and len(states)>1):
                    
                    #这里是在用valueagent估计每个state的rewards
                    states_tensor=torch.tensor(states, dtype=torch.float32).to(self.config.DEVICE)
                    #估计某个状态的价值
                    self.valueagent.eval()
                    #这states步模型估计的期望
                    print(states_tensor.shape)
                    values=self.valueagent(states_tensor).cpu().detach().numpy()
                    #未来一步的模型估计的期望
                    next_state_tensor=torch.tensor(next_state, dtype=torch.float32).to(self.config.DEVICE)
                    print(next_state_tensor.shape)
                    next_value=self.valueagent(next_state_tensor).cpu().detach().numpy()
                    
                    rewards=np.array(rewards)
                    #广义优势估计GAE
                    gaes=np.zeros_like(rewards)
                    #这几个状态的期望
                    n_steps_targets=np.zeros_like(rewards)
                    
                    gae_sum=0#对gae进行累积
                    forward_val=0#某步前面的期望
                    
                    if not done:#如果游戏没有结束的话,就用模型估计的期望,如果游戏结束了,那期望就为0
                        forward_val=next_value
                        
                    #这里从末尾开始计算,k步能够得到的rewards和后面有关    
                    for k in reversed(range(0, len(rewards))):
                        #第k步的rewards+未来的期望-当前步的期望 就是选择某步的优势
                        delta = rewards[k] + self.discount * forward_val - values[k]
                        #累加 k步以后的优势 +当前步的优势
                        gae_sum = self.discount * self.lmbda * gae_sum + delta
                        #第K步总的优势
                        gaes[k] = gae_sum
                        #k-1步的后面的期望就是k步的期望,这是模型预测出来的。
                        forward_val = values[k]
                        #第k步的优势+第k步的期望,期望就是走各步reward的平均
                        n_steps_targets[k] = gaes[k] + values[k]   
                        
                    old_policys=self.policyagent(states_tensor).detach()
                    
                    for epoch in range(epochs):
                        
                        #策略模型的优化
                        self.policyagent.train()
                        states_tensor=torch.tensor(states, dtype=torch.float32).to(self.config.DEVICE)
                        self.policyoptimizer.zero_grad()
                        #模型通过状态得到了决策的概率分布
                        new_policys=self.policyagent(states_tensor)
                        #变成了0~6
                        action_one_hot=F.one_hot(torch.Tensor(actions).long(),num_classes=7).detach().to(self.config.DEVICE)
                        old_p=torch.log(torch.sum(old_policys*action_one_hot))
                        new_p=torch.log(torch.sum(new_policys*action_one_hot))
                        ratio=torch.exp(old_p-new_p)
                        clip_ratio=torch.clip(ratio,1-self.clip_ratio,1+self.clip_ratio)
                        gaes_tensor=torch.tensor(gaes, dtype=torch.float32).detach().to(self.config.DEVICE)
                        policy_agent_loss=-torch.mean(torch.min(ratio*gaes_tensor,clip_ratio*gaes_tensor))   
                        policy_agent_loss.backward()
                        #优化器进行优化(梯度下降,降低误差)
                        self.policyoptimizer.step()
                        self.policyagent.eval()
                        
                        #价值期望模型的优化
                        #评估某个状态的期望模型的优化
                        self.valueagent.train()
                        states_tensor=torch.tensor(states, dtype=torch.float32).to(self.config.DEVICE)
                        self.valueoptimizer.zero_grad()
                        n_steps_targets_tensor=torch.tensor(n_steps_targets, dtype=torch.float32).to(self.config.DEVICE)
                        value_agent_loss=self.valueagent.loss_fn(n_steps_targets_tensor,self.valueagent(states_tensor))
                        value_agent_loss.backward()
                        #优化器进行优化(梯度下降,降低误差)
                        self.valueoptimizer.step()
                        self.valueagent.eval()
                    #训练完之后更新
                    one_game_rewards+=list(rewards)

                    # 随机删除
                    n = len(states)
                    probabilities = 0.8 * (1 - np.arange(n) / n)
                    random_values = np.random.rand(n)
                    mask = random_values > probabilities
                    

                    states=states[mask]#每个状态保存
                    actions=actions[mask]#每个状态后采取的action
                    rewards=rewards[mask]#采取action后得到的reward
                    old_policys=old_policys[mask]#每次action的概率分布
                if (done and len(states)==1):#游戏结束了,但是最后一个states没有训练
                    one_game_rewards+=list(rewards)
                    
                state=next_state

            total_rewards.append(one_game_rewards[-1])
            
            if (game_count%every==0) and (game_count):
                print(f"game_count:{game_count}")
                print(f"win_percent:{np.mean(np.array(total_rewards[game_count-every:game_count])==self.win_reward)}")
                print(f"draw_percent:{np.mean(np.array(total_rewards[game_count-every:game_count])==self.draw_reward)}")
                print(f"loss_percent:{np.mean(np.array(total_rewards[game_count-every:game_count])==self.loss_reward)}")
                print(f"error_percent:{np.mean(np.array(total_rewards[game_count-every:game_count])==self.error_reward)}")
                print(f"policy_agent_loss:{policy_agent_loss},value_agent_loss:{value_agent_loss}")
                print("-"*30)

                self.save()#保存模型
                self.to(self.config.DEVICE)