import numpy as np
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)

        self.max_size = size
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = np.array(obs).astype(np.float32)
        self.act_buf[self.ptr] = np.array(act).astype(np.float32)
        self.rew_buf[self.ptr] = np.array(rew).astype(np.float32)
        self.obs2_buf[self.ptr] = np.array(next_obs).astype(np.float32)
        self.done_buf[self.ptr] = np.array(done).astype(np.float32)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return dict(
            obs=self.obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            obs2=self.obs2_buf[idxs],
            done=self.done_buf[idxs],
        )

class WeightedReplayBuffer(ReplayBuffer):
    """Replay buffer that samples high-reward transitions more frequently."""
    def sample_weighted(self, batch_size, reward_bias=2.0):
        rewards = self.rew_buf[:self.size].flatten()
        r_i = np.abs(rewards) + 0.1
        weights = np.power(r_i, reward_bias)
        probabilitiess = weights / weights.sum()
        sampled_idxs = np.random.choice(np.arange(self.size), size=batch_size, p=probabilitiess, replace=True)
        res = {
            "obs":  self.obs_buf[sampled_idxs],
            "act":  self.act_buf[sampled_idxs],
            "rew":  self.rew_buf[sampled_idxs],
            "obs2": self.obs2_buf[sampled_idxs],
            "done": self.done_buf[sampled_idxs]
        }
        return res
