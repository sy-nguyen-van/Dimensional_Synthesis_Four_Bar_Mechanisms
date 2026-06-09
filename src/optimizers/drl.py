import torch
import numpy as np
from stable_baselines3 import PPO
from .drl_env import FourBarEnv

class DRLOptimizer:
    def __init__(self, objf, cons, LB, UB, popsize, totalgen, alpha, device=torch.device('cpu')):
        self.objf = objf
        self.cons = cons
        self.LB = LB
        self.UB = UB
        self.alpha = alpha
        self.device = device
        
        # Use totalgen * popsize as total timesteps to roughly match evaluation budget
        self.total_timesteps = totalgen * popsize
        
    def optimize(self):
        # Create environment
        # Each episode is roughly a single optimization trajectory of length 200
        env = FourBarEnv(
            objf=self.objf, 
            cons=self.cons, 
            LB=self.LB, 
            UB=self.UB, 
            alpha=self.alpha,
            max_steps=200, 
            device=self.device
        )
        
        # Initialize PPO model
        print(f"Training DRL Agent (PPO) for {self.total_timesteps} timesteps...")
        device_str = "cuda" if (hasattr(self.device, 'type') and self.device.type == 'cuda') else "cpu"
        model = PPO("MlpPolicy", env, verbose=0, device=device_str)
        
        # Train
        model.learn(total_timesteps=self.total_timesteps)
        
        print("Training complete. Running deterministic evaluation...")
        
        # Evaluate deterministic policy
        obs, _ = env.reset()
        best_x = env.state.copy()
        best_fval = env.best_fitness
        best_cost = best_fval
        best_cons = 0.0
        eval_history = []
        
        # Run one full episode with the trained deterministic policy
        for step in range(env.max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            eval_history.append(info['fitness'])
            
            if info['fitness'] < best_fval:
                best_fval = info['fitness']
                best_cost = info['cost']
                best_cons = info['constraint']
                best_x = info['raw_state']
                
            print(f"Eval Step {step+1}/{env.max_steps} | Cost: {info['cost']:.4f} | Constraint: {info['constraint']:.4f} | Fit: {info['fitness']:.4f}")
                
            if terminated or truncated:
                break
                
        # Return best parameters as tensor
        best_x_tensor = torch.tensor(best_x, dtype=torch.float32, device=self.device)
        
        return best_x_tensor, best_fval, eval_history
