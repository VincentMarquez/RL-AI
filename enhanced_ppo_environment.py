import numpy as np
import torch
from typing import Dict, Any, Tuple

class UniversalEnvironment:
    def __init__(self, state_dim: int, action_dim: int, resource_manager, ppo_bubble):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.resource_manager = resource_manager
        self.ppo = ppo_bubble
        self.max_steps = 200
        self.current_step = 0
        
    def reset(self, complexity: Dict[str, Any]) -> torch.Tensor:
        """Reset environment to initial state."""
        self.current_step = 0
        if self.resource_manager:
            state_dict = self.resource_manager.get_current_system_state()
            # Convert to tensor
            state_values = [
                state_dict.get('cpu_percent', 0) / 100.0,
                state_dict.get('memory_percent', 0) / 100.0,
                state_dict.get('energy', 5000) / 10000.0,
                state_dict.get('num_bubbles', 0) / 50.0,
            ]
            # Pad to state_dim
            while len(state_values) < self.state_dim:
                state_values.append(0.0)
            return torch.tensor(state_values[:self.state_dim], dtype=torch.float32)
        return torch.zeros(self.state_dim)
    
    async def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """Execute action and return next state, reward, done."""
        self.current_step += 1
        
        # Get current state
        if self.resource_manager:
            current_state = self.resource_manager.get_current_system_state()
        else:
            current_state = {}
        
        # Calculate reward based on system performance
        reward = self._calculate_reward(current_state, action)
        
        # Get next state
        next_state = self.reset({})  # Simplified
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        return next_state, reward, done
    
    def _calculate_reward(self, state: Dict, action: torch.Tensor) -> float:
        """Calculate reward based on system state."""
        reward = 0.0
        
        # Efficiency reward
        cpu = state.get('cpu_percent', 0)
        memory = state.get('memory_percent', 0)
        if cpu < 80 and memory < 80:
            reward += 0.3
        
        # Stability reward
        if state.get('error_rate', 0) < 0.1:
            reward += 0.3
        
        # Energy efficiency
        energy = state.get('energy', 0)
        if energy > 3000:
            reward += 0.2
        
        return np.clip(reward, -1.0, 1.0)
