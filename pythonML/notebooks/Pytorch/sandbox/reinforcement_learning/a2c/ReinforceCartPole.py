# get_ipython().system('pip install torch torchvision pyvirtualdisplay matplotlib seaborn pandas numpy pathlib gym')
# get_ipython().system('sudo apt-get install xvfb')
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random, os.path, math, glob, csv, base64, itertools, sys
import gym
from gym.wrappers import Monitor
from pprint import pprint


def make_seed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)



# ## REINFORCE

# ### Introduction
# 
# Reinforce is an actor-based **on policy** method. The policy $\pi_{\theta}$ is parametrized by a function approximator (e.g. a neural network).
# 
# Recall: $$ J(\pi) = \mathbb{E}_{\tau \sim \pi}\big[ \sum_{t} \gamma^t R_t \mid x_0, \pi \big].$$
# 
# To update the parameters $\theta$ of the policy, one has to do gradient ascent: $\theta_{k+1} = \theta_{k} + \alpha \nabla_{\theta}J(\pi_{\theta})|_{\theta_{k}}$.
# 
# **Advantages of this approach:**
# - Compared to a Q-learning approach, here the policy is directly parametrized so a small change of the parameters will not dramatically change the policy whereas this is not the case for Q-learning approaches.
# - The stochasticity of the policy allows exploration. In off policy learning, one has to deal with both a behaviour policy and an exploration policy.

# ### Policy Gradient Theorem
# 
# **Q.1: Prove the Policy Gradient Theorem:** $$ \nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)}\right]$$
# 
# 
# The policy gradient can be approximated with:
# $$ \hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau) $$

# #### Hint 1

# The probability of a trajectory $\tau = (s_{0}, a_{0},\dots, s_{T+1}$) with action chosen from $\displaystyle \pi_{\theta}$ is $P(\tau|\theta) = \rho_{0}(s_{0})\prod_{t=0}^{T}P\left(s_{t+1}|s_{t}, a_{t}\right) \pi_{\theta}(a_{t}|s_{t})$

# #### Hint 2

# Gradient-log trick: $  \nabla_{\theta}P(\tau|\theta)= P(\tau|\theta)\nabla_{\theta}\log P(\tau|\theta). $
# 
# The policy gradient can therefore be approximated with:
# $$ \hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau) $$

# ### Implementation of vanilla REINFORCE
# 
# **Q.2: Implement the REINFORCE algorithm**
# 
# The code is splitted in two parts:
# * The Model class defines the architecture of our neural network which takes as input the current state and returns the policy,
# * The Agent class is responsible for the training and evaluation procedure. You will need to code the method `optimize_model`.

# In[ ]:


class Model(nn.Module):
    def __init__(self, dim_observation, n_actions):
        super(Model, self).__init__()
        
        self.n_actions = n_actions
        self.dim_observation = dim_observation
        
        self.net = nn.Sequential(
            nn.Linear(in_features=self.dim_observation, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=self.n_actions),
            nn.Softmax(dim=0)
        )
        
    def forward(self, state):
        return self.net(state)
    
    def select_action(self, state):
        action = torch.multinomial(self.forward(state), 1)
        return action


# It is always nice to visualize the differents layers of our model.

# In[ ]:


env_id = 'CartPole-v1'
env = gym.make(env_id)
model = Model(env.observation_space.shape[0], env.action_space.n)
print(f'The model we created correspond to:\n{model}')


# We provide a base agent that you will need to extend in the next cell with your implementation of `optimize_model`.

# In[ ]:


class BaseAgent:
    
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config['env_id'])
        make_seed(config['seed'])
        self.env.seed(config['seed'])
        self.model = Model(self.env.observation_space.shape[0], self.env.action_space.n)
        self.gamma = config['gamma']
        self.optimizer = torch.optim.Adam(self.model.net.parameters(), lr=config['learning_rate'])
        self.monitor_env = Monitor(env, "./gym-results", force=True, video_callable=lambda episode: True)
    
    def _make_returns(self, rewards):
        """Returns the cumulative discounted rewards at each time step

        Parameters
        ----------
        rewards : array
            The array of rewards of one episode

        Returns
        -------
        array
            The cumulative discounted rewards at each time step
            
        Example
        -------
        for rewards=[1, 2, 3] this method outputs [1 + 2 * gamma + 3 * gamma**2, 2 + 3 * gamma, 3] 
        """
        
        returns = np.zeros_like(rewards)
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + self.gamma * returns[t + 1]
        return returns
    
    # Method to implement
    def optimize_model(self, n_trajectories):
        """Perform a gradient update using n_trajectories

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expectation card(D) in the formula above
        
        Returns
        -------
        array
            The cumulative discounted rewards of each trajectory
        """
        
        raise NotImplementedError
    
    def train(self, n_trajectories, n_update):
        """Training method

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expectation card(D) in the formula above
        n_update : int
            The number of gradient updates
            
        """
        
        rewards = []
        for episode in range(n_update):
            rewards.append(self.optimize_model(n_trajectories))
            print(f'Episode {episode + 1}/{n_update}: rewards {round(rewards[-1].mean(), 2)} +/- {round(rewards[-1].std(), 2)}')
        
        # Plotting
        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards[i]) for i in range(len(rewards))))), columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd');
        
    def evaluate(self):
        """Evaluate the agent on a single trajectory            
        """
        
        observation = self.monitor_env.reset()
        observation = torch.tensor(observation, dtype=torch.float)
        reward_episode = 0
        done = False
            
        while not done:
            action = self.model.select_action(observation)
            observation, reward, done, info = self.monitor_env.step(int(action))
            observation = torch.tensor(observation, dtype=torch.float)
            reward_episode += reward
        
        self.monitor_env.close()
        show_video("./gym-results")
        print(f'Reward: {reward_episode}')
        


# In[ ]:


class SimpleAgent(BaseAgent):
    
    def optimize_model(self, n_trajectories):
        """Perform a gradient update using n_trajectories

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expectation card(D) in the formula above
        
        Returns
        -------
        array
            The cumulative discounted rewards of each trajectory
        """
        
        ###
        # Your code here
        ###
        
        reward_trajectories = None
        loss = None
        
        # The following lines take care of the gradient descent step for the variable loss
        # that you need to compute.
        
        # Discard previous gradients
        self.optimizer.zero_grad()
        # Compute the gradient 
        loss.backward()
        # Do the gradient descent step
        self.optimizer.step()
        return reward_trajectories


# In the cell bellow are listed the parameters you should play with. Try out different configurations.

# In[ ]:


#@title Config {display-mode: "form", run: "auto"}

env_id = 'CartPole-v1'  #@param ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
learning_rate = 0.01  #@param {type: "number"}
gamma = 1  #@param {type: "number"}
seed = 1235  #@param {type: "integer"}
#@markdown ---

config = {
    'env_id': env_id,
    'learning_rate': learning_rate,
    'seed': seed,
    'gamma': gamma
}

print("Current config is:")
pprint(config)


# Let's train the agent.

# In[ ]:


agent = SimpleAgent(config)
agent.train(n_trajectories=50, n_update=50)


# Let's evaluate the quality of the learned policy.

# In[ ]:


agent.evaluate()


# **Q.3: What are the strengths and drawbacks of this algorithm? How would you improve it?**

# *Type your answer here*

# ### Don't let the past distract you

# - The sum of rewards during one episode has a high variance which affects the performance of this version of **REINFORCE**.
# - To assess the quality of an action, it make more sense to take into consideration only the rewards obtained after taking this action.
# - It can be proven that $$  \nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T \gamma^{t'-t} R(s_{t'}, a_{t'}, s_{t'+1})}\right].$$
# - **Bonus**: proof of this claim.
# - This has for effect to reduce the variance. Past rewards have zero mean but nonzero variance so they just add noise.  
# 
# **Q4: Implement this enhanced version of REINFORCE**

# In[ ]:


class EnhancedAgent(BaseAgent):
    
    def optimize_model(self, n_trajectories):
        """Perform a gradient update using n_trajectories

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expectation card(D) in the formula above
        
        Returns
        -------
        array
            The cumulative discounted rewards of each trajectory
        """
        
        ###
        # Your code here
        ###
        
        reward_trajectories = None
        loss = None
        
        self.optimizer.zero_grad()
        # Compute the gradient 
        loss.backward()
        # Do the gradient descent step
        self.optimizer.step()
        return reward_trajectories
   


# In[ ]:


#@title Config Enhanced {display-mode: "form", run: "auto"}

env_id = 'CartPole-v1'  #@param ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
learning_rate = 0.001  #@param {type: "number"}
gamma = 1  #@param {type: "number"}
seed = 1  #@param {type: "integer"}
#@markdown ---

config_enhanced = {
    'env_id': env_id,
    'learning_rate': learning_rate,
    'seed': seed,
    'gamma': gamma
}

print("Current config_enhanced is:")
pprint(config_enhanced)


# In[ ]:


agent = EnhancedAgent(config_enhanced)
agent.train(n_trajectories=50, n_update=50)


# In[ ]:


agent.evaluate()


# **Q.5: Did this method improve over vanilla REINFORCE?**

# *Type your answer here*

# ## A2C

# ### Theory
# The cumulative discounted reward has a high variance and therefore **REINFORCE** needs lots of trajectories to converge.
# In order to reduce it we can substract a baseline $b(s_t)$.
# 
# This is possible because:
# 
# $$\mathbb{E}_{a_t \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t|s_t) b(s_t)} = 0.$$
# 
# **Proof**
# 
# Let $P_\theta$ be the parameterized probability distribution over a random variable x.
# $$\int_x P_\theta (x) = 1$$
# Taking the gradient we get: $\nabla_\theta \int_x P_\theta (x) = \nabla_\theta 1 = 0$
# \begin{align*}
# 0 &= \nabla_\theta \int_x P_\theta (x)\\
# &= \int_x \nabla_\theta P_\theta(x)\\
# &= \int_x P_\theta (x) \nabla_\theta \log P_\theta (x) \\
# &= \mathbb{E}_{x \sim P_\theta} \nabla_\theta \log P_\theta (x)
# \end{align*}
# 
# Which leads to the following formula:
# 
# $$ \nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \left(\sum_{t'=t}^T \gamma^{t' - t} R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t)\right)}$$
# 
# The most common choice of baseline is the on-policy value function $V^{\pi}(s_t)$. This choice has the desirable effect of reducing variance in the sample estimate for the policy gradient. This results in faster and more stable policy learning. It is also appealing from a conceptual angle: it encodes the intuition that if an agent gets what it expected, it should “feel” neutral about it.
# 
# However $V^\pi$ is unknown and therefore we need to learn it. We will use a neural network to approximate $V^\pi$ with Mean Square Error as the loss function.
# 
# $$ \arg \min_{\phi} \mathbb{E}_{s_t, \hat{R}_t \sim \pi_k}{\left( V_{\phi}(s_t) - \hat{R}_t \right)^2}, $$
# 
# 
# We can show that we can replace $\sum_{t'=t}^T \gamma^{t' - t} R(s_{t'}, a_{t'}, s_{t'+1})$ in the formula above by $Q^{\pi_\theta}(s_t, a_t)$ and by doing so we finally get $$ \nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \left(Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t)\right)}$$
# 
# **Bonus: Prove the statement above**
# 
# $A(s_t, a_t) = Q(s_{t}, a_{t}) - V^\pi(s_t)$ is called the Advantage function which gives the name **Advantage Actor Critic**.
# 
# $Q^{\pi_\theta}(s_t, a_t)$ is approximated as the cumulative sum of rewards.
# 
# Minh et al. in their paper explained that they added an entropy term to the loss in order to encourage exploration.
# 
# $$ - \sum_{a} \pi(a | s) \log \pi(a | s) $$
# 
# **Q.6: Explain why adding the entropy term encourage exploration.**

# ### Coding
# 
# In the first part of this lab we had to wait for the end of an episode in order to compute the cumulative discounted rewards. Here we can use the critic to estimate the cumulative discounted reward and therefore we no longer need to wait for the episode termination.
# 
# **Example**: For a trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_t, a_t, r_t, ..., s_T, a_T, r_T)$ the cumulative discounted rewards $R(\tau) = \sum_{i=0}^T \gamma^i r_i$ can be approximated by $ \sum_{i=0}^t \gamma^i r_i + \gamma^{t+1} V_\phi(s_{t+1})$ where $V_\phi$ is our critic.
# 
# This is allows us to train our model using batch data, which is more efficient for:
# * Computing: deep learning libraries (PyTorch, TensorFlow, ...) are optimized for batched data;
# * We don't have to wait for the end of a long episode in order to perform the update;
# * It is more sample efficient;
# * ...
# 
# 
# **Q.7: Implement `optimize_model` using batches**

# In[3]:


class ActorNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out), dim=-1)
        return out
    
    def select_action(self, x):
        return torch.multinomial(self(x), 1).detach().numpy()


# In[4]:


class ValueNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


# In[5]:


class A2CAgent:

    def __init__(self, config):
        self.config = config
        self.env = gym.make(config['env_id'])
        make_seed(config['seed'])
        self.env.seed(config['seed'])
        self.monitor_env = Monitor(self.env, "./gym-results", force=True, video_callable=lambda episode: True)
        self.gamma = config['gamma']
        
        # Our two networks
        self.value_network = ValueNetwork(self.env.observation_space.shape[0], 16, 1)
        self.actor_network = ActorNetwork(self.env.observation_space.shape[0], 16, self.env.action_space.n)
        
        # Their optimizers
        self.value_network_optimizer = optim.RMSprop(self.value_network.parameters(), lr=config['value_network']['learning_rate'])
        self.actor_network_optimizer = optim.RMSprop(self.actor_network.parameters(), lr=config['actor_network']['learning_rate'])
        
    # Hint: use it during training_batch
    def _returns_advantages(self, rewards, dones, values, next_value):
        """Returns the cumulative discounted rewards at each time step

        Parameters
        ----------
        rewards : array
            An array of shape (batch_size,) containing the rewards given by the env
        dones : array
            An array of shape (batch_size,) containing the done bool indicator given by the env
        values : array
            An array of shape (batch_size,) containing the values given by the value network
        next_value : float
            The value of the next state given by the value network
        
        Returns
        -------
        returns : array
            The cumulative discounted rewards
        advantages : array
            The advantages
        """
        
        returns = np.append(np.zeros_like(rewards), [next_value], axis=0)
        
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
            
        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def training_batch(self, epochs, batch_size):
        """Perform a training by batch

        Parameters
        ----------
        epochs : int
            Number of epochs
        batch_size : int
            The size of a batch
        """
        episode_count = 0
        actions = np.empty((batch_size,), dtype=np.int)
        dones = np.empty((batch_size,), dtype=np.bool)
        rewards, values = np.empty((2, batch_size), dtype=np.float)
        observations = np.empty((batch_size,) + self.env.observation_space.shape, dtype=np.float)
        observation = self.env.reset()
        rewards_test = []

        for epoch in range(epochs):
            # Lets collect one batch
            for i in range(batch_size):
                observations[i] = observation
                values[i] = self.value_network(torch.tensor(observation, dtype=torch.float)).detach().numpy()
                policy = self.actor_network(torch.tensor(observation, dtype=torch.float))
                actions[i] = torch.multinomial(policy, 1).detach().numpy()
                observation, rewards[i], dones[i], _ = self.env.step(actions[i])

                if dones[i]:
                    observation = self.env.reset()

            # If our epiosde didn't end on the last step we need to compute the value for the last state
            if dones[-1]:
                next_value = 0
            else:
                next_value = self.value_network(torch.tensor(observation, dtype=torch.float)).detach().numpy()[0]
            
            # Update episode_count
            episode_count += sum(dones)

            # Compute returns and advantages
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)

            # Learning step !
            self.optimize_model(observations, actions, returns, advantages)

            # Test it every 50 epochs
            if epoch % 50 == 0 or epoch == epochs - 1:
                rewards_test.append(np.array([self.evaluate() for _ in range(50)]))
                print(f'Epoch {epoch}/{epochs}: Mean rewards: {round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}')

                # Early stopping
                if rewards_test[-1].mean() > 490 and epoch != epochs -1:
                    print('Early stopping !')
                    break
                observation = self.env.reset()
                    
        # Plotting
        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards_test[i]) for i in range(len(rewards_test))))), columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd');
        
        print(f'The trainnig was done over a total of {episode_count} episodes')

    def optimize_model(self, observations, actions, returns, advantages):
        actions = F.one_hot(torch.tensor(actions), self.env.action_space.n)
        returns = torch.tensor(returns[:, None], dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)
        observations = torch.tensor(observations, dtype=torch.float)

        # MSE for the values
        # Actor & Entropy loss

        # MSE for the values
        self.value_network_optimizer.zero_grad()
        values = self.value_network(observations)
        loss_value = 1 * F.mse_loss(values, returns)
        loss_value.backward()
        self.value_network_optimizer.step()

        # Actor loss
        self.actor_network_optimizer.zero_grad()
        policies = self.actor_network(observations)
        loss_policy = ((actions.float() * policies.log()).sum(-1) * advantages).mean()
        loss_entropy = - (policies * policies.log()).sum(-1).mean()
        loss_actor = - loss_policy - 0.0001 * loss_entropy
        loss_actor.backward()
        self.actor_network_optimizer.step()
        
        return loss_value, loss_actor      
       

    def evaluate(self, render=False):
        env = self.monitor_env if render else self.env
        observation = env.reset()
        observation = torch.tensor(observation, dtype=torch.float)
        reward_episode = 0
        done = False

        while not done:
            policy = self.actor_network(observation)
            action = torch.multinomial(policy, 1)
            observation, reward, done, info = env.step(int(action))
            observation = torch.tensor(observation, dtype=torch.float)
            reward_episode += reward
            
        env.close()
        if render:
            show_video("./gym-results")
            print(f'Reward: {reward_episode}')
        return reward_episode


# **Q.8: Try out different hyperparameters (batch size, learning rate, optimizer, gamma) and identify how each one of them influence the learning.**

# In[15]:


#@title Config A2C {display-mode: "form", run: "auto"}

env_id = 'Acrobot-v1'  #@param ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
value_learning_rate = 0.001  #@param {type: "number"}
actor_learning_rate = 0.001  #@param {type: "number"}
gamma = 1  #@param {type: "number"}
entropy = 1  #@param {type: "number"}
seed = 1  #@param {type: "integer"}
#@markdown ---

config_a2c = {
    'env_id': env_id,
    'gamma': gamma,
    'seed': seed,
    'value_network': {'learning_rate': value_learning_rate},
    'actor_network': {'learning_rate': actor_learning_rate},
    'entropy': entropy
}

print("Current config_a2c is:")
pprint(config_a2c)


# In[16]:


agent = A2CAgent(config_a2c)
rewards = agent.training_batch(1000, 256)


# In[17]:


agent.evaluate(True);


# **Q.9: What are the strengths and drawbacks of this algorithm? How would you improve it?**

# *Type your answer here*

# **Q.10: Compare the three algorithms (sample efficiency, stability, ...)**

# *Type your answer here*

# Make sure to try the algorithms with other environments!
