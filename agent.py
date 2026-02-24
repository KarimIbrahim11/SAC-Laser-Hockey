import copy
import random
import torch
from torch.distributions import Normal
import submodules.hockey_env.hockey.hockey_env as h_env
import numpy as np
from gymnasium import spaces


import memory as mem
from feedforward import Feedforward, MultiHeadFeedforward
from obs_scaling import RunningMeanStd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)


_LOG_STD_MAX = 2
_LOG_STD_MIN = -20
_NUM_STABILITY_EPSILON = 1e-6
_DEBUG_MODE = True

class SelfPlayOpponent(object):
    def __init__(self, agent):
        self.agent = copy.deepcopy(agent)
        self.agent.actor_policy.eval()
    def act(self, obs):
        obs_scaled = self.agent.obs_normalizer.normalize(obs)
        action = self.agent.act(obs_scaled, deterministic=True)
        return action
    

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible"""
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class DCQ(torch.nn.Module):
    """Double Clipped Q learning with strict value bounds."""
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100],
                 learning_rate=0.0002, q_min=-10.0, q_max=10.0):
        super().__init__()
        self.Q1_function = Feedforward(input_size=observation_dim + action_dim, 
                                      hidden_sizes=hidden_sizes, output_size=1,
                                      activation_fun=torch.nn.Tanh(),
                                      use_layernorm=True)
        self.Q2_function = Feedforward(input_size=observation_dim + action_dim, 
                                      hidden_sizes=hidden_sizes, output_size=1,
                                      activation_fun=torch.nn.Tanh(),
                                      use_layernorm=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-6)
        self.loss = torch.nn.SmoothL1Loss()
        
        self.q_min = q_min
        self.q_max = q_max
    
    def forward(self, observations, actions):
        input_tensor = torch.cat([observations, actions], dim=1)
        q1 = self.Q1_function(input_tensor)
        q2 = self.Q2_function(input_tensor)
        
        q1 = torch.clamp(q1, self.q_min, self.q_max)
        q2 = torch.clamp(q2, self.q_min, self.q_max)
        return q1, q2
    
    def fit(self, observations, actions, targets):
        self.train()
        
        targets = torch.clamp(targets, self.q_min, self.q_max)
        
        self.optimizer.zero_grad()
        q1_pred, q2_pred = self.forward(observations, actions)
        
        loss1 = self.loss(q1_pred, targets)
        loss2 = self.loss(q2_pred, targets)
        total_loss = loss1 + loss2
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item()

class SACAgent(object):
    """Agent implementing SAC algorithm with NN function approximation."""
    def __init__(self, observation_space, action_space, **userconfig):
        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible with {}. (Require: Box)'.format(
                observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}. (Require Box)'.format(
                action_space, self))

        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        
        self.obs_normalizer = RunningMeanStd(shape=(self._obs_dim,))
        self.reward_normalizer = RunningMeanStd(shape=(1,))
        self.normalize_rewards = True
        self._config = {
            "discount": 0.99,
            "buffer_size": int(1e6),
            "batch_size": 512,
            "learning_rate_actor": 5e-5,
            "learning_rate_critic": 1e-4,
            "hidden_sizes_actor": [256, 256],
            "hidden_sizes_critic": [256, 256],
            "tau": 0.005,
            "auto_alpha": True,
            "initial_alpha": 0.3,  #0.2
            "reward_scale": 1.0,
            "q_min": -10.0,
            "q_max": 10.0,
        }
        self._config.update(userconfig)
        
        # Automatic alpha tuning
        init_log_val = float(np.log(self._config["initial_alpha"]))
        self.log_alpha = torch.tensor([init_log_val], 
                                     requires_grad=self._config["auto_alpha"], 
                                     device=device,
                                     dtype=torch.float32)
        
        if self._config["auto_alpha"]:
            self.target_entropy = -float(self._action_n)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], 
                                                   lr=float(5e-5),
                                                   eps=1e-6)
        else:
            self.alpha_optimizer = None
            self.target_entropy = None

        self.buffer = mem.WeightedReplayBuffer(
            obs_dim=self._obs_dim,
            act_dim=self._action_n,
            size=self._config["buffer_size"]
        )
        # self.success_buffer = mem.ReplayBuffer(
        #     obs_dim=self._obs_dim,
        #     act_dim=self._action_n,
        #     size=int(1e5)  # Smaller buffer just for wins/good episodes
        # )
        # self.use_success_buffer = True
        # self.success_replay_ratio = 0.3

        # DCQ Critic Networks
        self.critic = DCQ(
            observation_dim=self._obs_dim,
            action_dim=self._action_n,
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=self._config["learning_rate_critic"],
            q_min=self._config["q_min"],
            q_max=self._config["q_max"]
        )
        self.critic_target = DCQ(
            observation_dim=self._obs_dim,
            action_dim=self._action_n,
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=self._config["learning_rate_critic"],
            q_min=self._config["q_min"],
            q_max=self._config["q_max"]
        )
        
        # Actor Policy Network
        self.actor_policy = MultiHeadFeedforward(input_size=self._obs_dim, 
                                                output_size=self._action_n, 
                                                hidden_sizes=self._config["hidden_sizes_actor"],
                                                activation_fun=torch.nn.Tanh(),
                                                use_layernorm=True)
        
        self.critic.to(device)
        self.critic_target.to(device)
        self.actor_policy.to(device)
        self._copy_nets()

        self.optimizer = torch.optim.Adam(self.actor_policy.parameters(),
                                         lr=self._config["learning_rate_actor"],
                                         eps=1e-6)
        self.train_iter = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _copy_nets(self):
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def _soft_update(self):
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), 
                                          self.critic_target.parameters()):
                target_param.data.copy_(
                    self._config['tau'] * param.data + 
                    (1.0 - self._config['tau']) * target_param.data
                )

    def act(self, observations, deterministic=True):
        if not isinstance(observations, torch.Tensor):
            observations = torch.as_tensor(observations, dtype=torch.float32, device=device)
        
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)

        with torch.no_grad():
            mu, log_sigma = self.actor_policy(observations)
            
            if deterministic:
                action = torch.tanh(mu)
            else:
                std = torch.exp(torch.clamp(log_sigma, _LOG_STD_MIN, _LOG_STD_MAX))
                dist = torch.distributions.Normal(mu, std)
                u = dist.sample()
                action = torch.tanh(u)
            
            action = action.cpu().numpy().reshape(-1)
            low, high = self._action_space.low, self._action_space.high
            scaled_action = low + (action + 1.0) * 0.5 * (high - low)
            clipped_action = np.clip(scaled_action, low, high)
        return clipped_action

    def sample(self, observations):
        # using rsample for parametrization trick for training
        mu, log_sigma = self.actor_policy(observations)
        std = torch.exp(torch.clamp(log_sigma, _LOG_STD_MIN, _LOG_STD_MAX))
        dist = Normal(mu, std)
        u = dist.rsample()
        # TODO: Think about the tanh to be between high and low
        actions = torch.tanh(u)
        log_prob_u = dist.log_prob(u)
        correction = torch.log(1- actions.pow(2) + _NUM_STABILITY_EPSILON)
        log_probs = (log_prob_u - correction).sum(dim=-1, keepdim=True)
        
        dbg = {}
        if _DEBUG_MODE:
            with torch.no_grad():
                # Saturation: % of mu values pushed to the flat part of tanh (>2.0)
                dbg["sat"] = (torch.abs(mu) > 2.0).float().mean().item() * 100
                dbg["std"] = std.mean().item()
                dbg["mu_max"] = torch.abs(mu).max().item()
                dbg["mu_avg"] = torch.abs(mu).mean().item()
        
        return actions, log_probs, dbg
          
    def store_transition(self, transition):
        ob, action, reward, new_ob, done = transition
        self.buffer.add(ob, action, reward, new_ob, done)

    def state(self):
        return {
            'actor': self.actor_policy.state_dict(),
            'critic': self.critic.state_dict(),
            'log_alpha': self.log_alpha,
            'obs_normalizer': {
                'mean': self.obs_normalizer.mean,
                'var': self.obs_normalizer.var,
                'count': self.obs_normalizer.count
            }
        }

    def restore_state(self, state_dict):
        self.actor_policy.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.log_alpha.data.copy_(state_dict['log_alpha'].data)
        self._copy_nets()
        if 'obs_normalizer' in state_dict:
            obs_state = state_dict['obs_normalizer']
            self.obs_normalizer.mean = obs_state['mean']
            self.obs_normalizer.var = obs_state['var']
            self.obs_normalizer.count = obs_state['count']
            print("Loaded observation scaling statistics.")

    def train(self, iter_fit=1, i_episode=None):
        train_stats = {
            "q_loss": [], 
            "actor_loss": [], 
            "alpha_loss": [], 
            "alpha": [], 
            "entropy": [],
            "q_mean": [],
            "q_std": [],
            "target_mean": []
        }
        
        for fit_step in range(iter_fit):
            batch = self.buffer.sample(self._config["batch_size"])
            
            s = torch.as_tensor(batch["obs"], device=device, dtype=torch.float32)
            a = torch.as_tensor(batch["act"], device=device, dtype=torch.float32)
            rew = torch.as_tensor(batch["rew"], device=device, dtype=torch.float32)
            s_prime = torch.as_tensor(batch["obs2"], device=device, dtype=torch.float32)
            done = torch.as_tensor(batch["done"], device=device, dtype=torch.float32)
            
            # NaN check on inputs
            if torch.isnan(s).any() or torch.isnan(a).any() or torch.isnan(rew).any():
                print("NaN in batch! Skipping.")
                continue
            
            with torch.no_grad():
                a_prime, log_prob_prime, _ = self.sample(s_prime)
                
                q1_target, q2_target = self.critic_target(s_prime, a_prime)
                q_target_min = torch.min(q1_target, q2_target)
                
                if torch.isnan(q_target_min).any():
                    print("NaN in target Q! Resetting target network.")
                    self._copy_nets()
                    continue
                
                target = rew + self._config['discount'] * (1 - done) * (
                    q_target_min - self.alpha.detach() * log_prob_prime
                )
                
                target = torch.clamp(target, self._config['q_min'], self._config['q_max'])
            
            q_loss_value = self.critic.fit(s, a, target)
            
            if np.isnan(q_loss_value):
                print("NaN in Q-loss! Skipping update.")
                continue
            
            # dbg is for debugging purposes
            a_curr, log_p_curr, dbg = self.sample(s)
            
            q1_curr, q2_curr = self.critic(s, a_curr)
            q_min_curr = torch.min(q1_curr, q2_curr)
            
            # Actor objective
            actor_loss = (self.alpha.detach() * log_p_curr - q_min_curr).mean()
            
            if torch.isnan(actor_loss):
                print("NaN in actor loss! Skipping.")
                continue
            
            self.optimizer.zero_grad()
            actor_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_policy.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            
            # alpha auto-tunign
            if self._config["auto_alpha"]:
                alpha_loss = -(self.log_alpha * (log_p_curr + self.target_entropy).detach()).mean()
                
                if not torch.isnan(alpha_loss):
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    torch.nn.utils.clip_grad_value_(self.log_alpha, clip_value=1.0)
                    self.alpha_optimizer.step()
                    
                    train_stats["alpha_loss"].append(alpha_loss.item())
                else:
                    train_stats["alpha_loss"].append(0.0)
            else:
                train_stats["alpha_loss"].append(0.0)
        
            
            with torch.no_grad():
                q_mean = q_min_curr.mean().item()
                q_std = q_min_curr.std().item()
                target_mean = target.mean().item()
                
                train_stats["q_loss"].append(q_loss_value)
                train_stats["actor_loss"].append(actor_loss.item())
                train_stats["alpha"].append(self.alpha.item())
                train_stats["entropy"].append(-log_p_curr.mean().item())
                train_stats["q_mean"].append(q_mean)
                train_stats["q_std"].append(q_std)
                train_stats["target_mean"].append(target_mean)

            self._soft_update()
        
        return {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in train_stats.items()}