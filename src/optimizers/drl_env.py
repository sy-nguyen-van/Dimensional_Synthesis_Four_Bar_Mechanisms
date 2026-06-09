import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

class FourBarEnv(gym.Env):
    """
    Gymnasium environment for Four-Bar Mechanism Synthesis.
    The agent iteratively adjusts the parameters to minimize the cost function.
    """
    def __init__(self, objf, cons, LB, UB, alpha, max_steps=100, device='cpu'):
        super().__init__()
        self.objf = objf
        self.cons = cons
        self.LB = LB.cpu().numpy()
        self.UB = UB.cpu().numpy()
        self.alpha = alpha
        self.nvars = len(self.LB)
        self.max_steps = max_steps
        self.device = device
        
        # Action space: continuous adjustments between -1 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nvars,), dtype=np.float32)
        
        # Observation space: normalized current parameters [-1, 1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nvars,), dtype=np.float32)
        
        self.state = None
        self.current_step = 0
        self.best_fitness = float('inf')
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Random initialization
        self.state = np.random.uniform(self.LB, self.UB).astype(np.float32)
        
        # Calculate initial fitness
        x_tensor = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
        f = self.objf(x_tensor).item()
        V = self.cons(x_tensor).item()
        self.best_fitness = f + self.alpha * V
        
        return self._normalize(self.state), {}
        
    def step(self, action):
        self.current_step += 1
        
        # Denormalize action to get parameter changes
        # Max step size is 10% of the bounds range per step
        max_delta = (self.UB - self.LB) * 0.1
        delta = action * max_delta
        
        # Update state and clip to bounds
        self.state = np.clip(self.state + delta, self.LB, self.UB).astype(np.float32)
        
        # Evaluate new state
        x_tensor = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
        f = self.objf(x_tensor).item()
        V = self.cons(x_tensor).item()
        fitness = f + self.alpha * V
        
        # Reward is improvement in fitness
        if fitness < self.best_fitness:
            reward = self.best_fitness - fitness
            self.best_fitness = fitness
        else:
            reward = -0.1  # Small penalty for no improvement
            
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        info = {
            'cost': f,
            'constraint': V,
            'fitness': fitness,
            'raw_state': self.state.copy()
        }
        
        return self._normalize(self.state), reward, terminated, truncated, info
        
    def _normalize(self, state):
        # Normalize to [-1, 1]
        normalized = 2.0 * (state - self.LB) / (self.UB - self.LB + 1e-8) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
