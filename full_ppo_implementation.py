"""
COMPLETE PPO IMPLEMENTATION WITH HYPERPARAMETER TUNING AND ADVANCED REPLAY BUFFERS
Ready for immediate use in your system
"""

# ==================== FILE 1: advanced_replay_buffers.py ====================

import numpy as np
import torch
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any, Union
import heapq
import random
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)

# Data structures
Transition = namedtuple('Transition', 
    ['state', 'action', 'reward', 'next_state', 'done', 'info'])

@dataclass
class Experience:
    """Enhanced experience with additional metadata."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any]
    td_error: float = 0.0
    episode_id: int = 0
    step_id: int = 0
    timestamp: float = 0.0


class BaseReplayBuffer(ABC):
    """Base class for all replay buffers."""
    
    def __init__(self, capacity: int, device: str = 'cpu'):
        self.capacity = capacity
        self.device = device
        self.size = 0
        
    @abstractmethod
    def push(self, *args):
        """Add experience to buffer."""
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> Any:
        """Sample batch from buffer."""
        pass
    
    def __len__(self):
        return self.size


class PrioritizedReplayBuffer(BaseReplayBuffer):
    """
    Prioritized Experience Replay Buffer
    Samples experiences based on TD error priority
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001, device: str = 'cpu'):
        super().__init__(capacity, device)
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # Storage
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
        logger.info(f"PrioritizedReplayBuffer initialized: capacity={capacity}, alpha={alpha}, beta={beta}")
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool, info: Dict = None):
        """Add experience with max priority."""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {},
            td_error=self.max_priority,
            timestamp=time.time()
        )
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Set max priority for new experience
        self.priorities[self.position] = self.max_priority ** self.alpha
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, beta: float = None) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights."""
        if beta is None:
            beta = self.beta
            self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD errors."""
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.buffer[idx].td_error = td_errors[idx]
        
        self.max_priority = max(self.max_priority, priorities.max())
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        priorities = self.priorities[:self.size]
        return {
            'size': self.size,
            'capacity': self.capacity,
            'beta': self.beta,
            'max_priority': self.max_priority,
            'mean_priority': priorities.mean(),
            'std_priority': priorities.std()
        }


class HindsightExperienceReplayBuffer(BaseReplayBuffer):
    """
    Hindsight Experience Replay for goal-conditioned tasks
    """
    
    def __init__(self, capacity: int, k: int = 4, strategy: str = 'future',
                 device: str = 'cpu'):
        super().__init__(capacity, device)
        self.k = k  # Number of hindsight goals per transition
        self.strategy = strategy  # 'future', 'episode', 'random'
        self.episodes = []
        self.current_episode = []
        
        logger.info(f"HER Buffer initialized: capacity={capacity}, k={k}, strategy={strategy}")
    
    def start_episode(self):
        """Start collecting a new episode."""
        self.current_episode = []
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool, goal: np.ndarray, 
             achieved_goal: np.ndarray, info: Dict = None):
        """Add experience with goal information."""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'goal': goal,
            'achieved_goal': achieved_goal,
            'next_achieved_goal': info.get('next_achieved_goal', achieved_goal),
            'info': info or {}
        }
        
        self.current_episode.append(experience)
        
        if done:
            self._process_episode()
    
    def _process_episode(self):
        """Process episode and generate hindsight experiences."""
        if not self.current_episode:
            return
        
        # Store original episode
        self.episodes.append(list(self.current_episode))
        
        # Generate hindsight experiences
        for t, transition in enumerate(self.current_episode):
            # Store original transition
            self._store_transition(transition)
            
            # Generate k hindsight goals
            for _ in range(self.k):
                her_transition = self._generate_hindsight_transition(t)
                if her_transition:
                    self._store_transition(her_transition)
        
        # Manage capacity
        while sum(len(ep) for ep in self.episodes) > self.capacity:
            self.episodes.pop(0)
        
        self.current_episode = []
    
    def _generate_hindsight_transition(self, t: int) -> Optional[Dict]:
        """Generate single hindsight transition."""
        episode = self.current_episode
        
        if self.strategy == 'future':
            # Sample from future states in episode
            if t >= len(episode) - 1:
                return None
            future_idx = np.random.randint(t + 1, len(episode))
            new_goal = episode[future_idx]['achieved_goal']
            
        elif self.strategy == 'episode':
            # Sample from entire episode
            ep_idx = np.random.randint(0, len(episode))
            new_goal = episode[ep_idx]['achieved_goal']
            
        elif self.strategy == 'random':
            # Sample from any episode
            if not self.episodes:
                return None
            rand_ep = random.choice(self.episodes)
            rand_idx = np.random.randint(0, len(rand_ep))
            new_goal = rand_ep[rand_idx]['achieved_goal']
        
        else:
            raise ValueError(f"Unknown HER strategy: {self.strategy}")
        
        # Create hindsight transition
        transition = episode[t].copy()
        transition['goal'] = new_goal
        transition['reward'] = self._compute_reward(
            transition['achieved_goal'], 
            new_goal,
            transition['info']
        )
        
        return transition
    
    def _compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, 
                       info: Dict) -> float:
        """Compute reward for achieved vs desired goal."""
        # Simple L2 distance reward (customize for your task)
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return -1.0 if distance > 0.05 else 0.0
    
    def _store_transition(self, transition: Dict):
        """Store processed transition."""
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample batch from all episodes."""
        all_transitions = []
        for episode in self.episodes:
            all_transitions.extend(episode)
        
        if len(all_transitions) < batch_size:
            return random.choices(all_transitions, k=batch_size)
        
        return random.sample(all_transitions, batch_size)


class NStepReplayBuffer(BaseReplayBuffer):
    """
    N-step replay buffer for multi-step returns
    """
    
    def __init__(self, capacity: int, n_steps: int = 3, gamma: float = 0.99,
                 device: str = 'cpu'):
        super().__init__(capacity, device)
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_steps)
        self.buffer = deque(maxlen=capacity)
        
        logger.info(f"N-Step Buffer initialized: capacity={capacity}, n_steps={n_steps}, gamma={gamma}")
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool, info: Dict = None):
        """Add experience and compute n-step returns."""
        self.n_step_buffer.append(Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {},
            timestamp=time.time()
        ))
        
        # Only process when we have enough steps or episode ends
        if len(self.n_step_buffer) == self.n_steps or done:
            self._compute_n_step_transition()
        
        # Clear buffer on episode end
        if done:
            while len(self.n_step_buffer) > 0:
                self._compute_n_step_transition()
    
    def _compute_n_step_transition(self):
        """Compute n-step return and create transition."""
        if not self.n_step_buffer:
            return
        
        # Get first transition
        first = self.n_step_buffer[0]
        
        # Compute n-step return
        n_step_return = 0.0
        gamma_power = 1.0
        
        for i, exp in enumerate(self.n_step_buffer):
            n_step_return += gamma_power * exp.reward
            gamma_power *= self.gamma
            
            if exp.done:
                break
        
        # Get final state
        last = self.n_step_buffer[-1]
        
        # Create n-step transition
        n_step_exp = Experience(
            state=first.state,
            action=first.action,
            reward=n_step_return,
            next_state=last.next_state,
            done=last.done,
            info={
                'n_steps': len(self.n_step_buffer),
                'original_reward': first.reward,
                **first.info
            },
            timestamp=first.timestamp
        )
        
        self.buffer.append(n_step_exp)
        self.size = len(self.buffer)
        self.n_step_buffer.popleft()
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of n-step transitions."""
        if self.size < batch_size:
            return list(self.buffer)
        
        return random.sample(self.buffer, batch_size)


class CombinedReplayBuffer(BaseReplayBuffer):
    """
    Combines multiple replay strategies (uniform, prioritized, rare events)
    """
    
    def __init__(self, capacity: int, rare_threshold: float = 0.1, 
                 buffer_ratios: Dict[str, float] = None, device: str = 'cpu'):
        super().__init__(capacity, device)
        
        # Default ratios
        if buffer_ratios is None:
            buffer_ratios = {
                'uniform': 0.5,
                'prioritized': 0.3,
                'rare': 0.2
            }
        
        self.buffer_ratios = buffer_ratios
        self.rare_threshold = rare_threshold
        
        # Sub-buffers
        buffer_sizes = {
            name: int(capacity * ratio) 
            for name, ratio in buffer_ratios.items()
        }
        
        self.uniform_buffer = deque(maxlen=buffer_sizes['uniform'])
        self.prioritized_buffer = PrioritizedReplayBuffer(
            buffer_sizes['prioritized'], device=device
        )
        self.rare_buffer = deque(maxlen=buffer_sizes['rare'])
        
        logger.info(f"Combined Buffer initialized with ratios: {buffer_ratios}")
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool, info: Dict = None):
        """Route experience to appropriate sub-buffer."""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {},
            timestamp=time.time()
        )
        
        # Always add to uniform buffer
        self.uniform_buffer.append(experience)
        
        # Add to prioritized buffer
        self.prioritized_buffer.push(state, action, reward, next_state, done, info)
        
        # Check if rare event
        is_rare = self._is_rare_event(experience)
        if is_rare:
            self.rare_buffer.append(experience)
        
        self.size = len(self.uniform_buffer) + len(self.prioritized_buffer) + len(self.rare_buffer)
    
    def _is_rare_event(self, experience: Experience) -> bool:
        """Determine if experience is rare/important."""
        # High reward
        if abs(experience.reward) > 1.0:
            return True
        
        # Terminal state with unusual outcome
        if experience.done and experience.reward < -0.5:
            return True
        
        # Custom criteria from info
        if experience.info.get('is_rare', False):
            return True
        
        # Random rare sampling
        return random.random() < self.rare_threshold
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], Dict[str, int]]:
        """Sample from multiple buffers according to ratios."""
        samples = []
        source_counts = {'uniform': 0, 'prioritized': 0, 'rare': 0}
        
        for buffer_name, ratio in self.buffer_ratios.items():
            n_samples = int(batch_size * ratio)
            
            if buffer_name == 'uniform' and len(self.uniform_buffer) > 0:
                uniform_samples = random.sample(
                    self.uniform_buffer, 
                    min(n_samples, len(self.uniform_buffer))
                )
                samples.extend(uniform_samples)
                source_counts['uniform'] = len(uniform_samples)
                
            elif buffer_name == 'prioritized' and len(self.prioritized_buffer) > 0:
                per_samples, _, _ = self.prioritized_buffer.sample(
                    min(n_samples, len(self.prioritized_buffer))
                )
                samples.extend(per_samples)
                source_counts['prioritized'] = len(per_samples)
                
            elif buffer_name == 'rare' and len(self.rare_buffer) > 0:
                rare_samples = random.sample(
                    self.rare_buffer,
                    min(n_samples, len(self.rare_buffer))
                )
                samples.extend(rare_samples)
                source_counts['rare'] = len(rare_samples)
        
        # Fill remaining with uniform sampling if needed
        remaining = batch_size - len(samples)
        if remaining > 0 and len(self.uniform_buffer) > 0:
            extra_samples = random.sample(
                self.uniform_buffer,
                min(remaining, len(self.uniform_buffer))
            )
            samples.extend(extra_samples)
            source_counts['uniform'] += len(extra_samples)
        
        return samples, source_counts


class ReplayBufferWrapper:
    """Wrapper to integrate replay buffers with PPO."""
    
    def __init__(self, buffer_type: str = 'prioritized', capacity: int = 10000, 
                 device: str = 'cpu', **kwargs):
        self.buffer_type = buffer_type
        self.device = device
        
        # Create appropriate buffer
        if buffer_type == 'prioritized':
            self.buffer = PrioritizedReplayBuffer(capacity, device=device, **kwargs)
        elif buffer_type == 'her':
            self.buffer = HindsightExperienceReplayBuffer(capacity, device=device, **kwargs)
        elif buffer_type == 'nstep':
            self.buffer = NStepReplayBuffer(capacity, device=device, **kwargs)
        elif buffer_type == 'combined':
            self.buffer = CombinedReplayBuffer(capacity, device=device, **kwargs)
        else:
            raise ValueError(f"Unknown buffer type: {buffer_type}")
        
        logger.info(f"ReplayBufferWrapper created with {buffer_type} buffer")
    
    def store_rollout(self, states: torch.Tensor, actions: torch.Tensor, 
                     rewards: torch.Tensor, next_states: torch.Tensor, 
                     dones: torch.Tensor, infos: List[Dict] = None):
        """Store entire rollout in buffer."""
        batch_size = states.shape[0]
        
        for i in range(batch_size):
            self.buffer.push(
                states[i].cpu().numpy(),
                actions[i].cpu().numpy(),
                rewards[i].item(),
                next_states[i].cpu().numpy(),
                dones[i].item(),
                infos[i] if infos else {}
            )
    
    def sample_for_ppo(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch formatted for PPO training."""
        if self.buffer_type == 'prioritized':
            experiences, indices, weights = self.buffer.sample(batch_size)
            
            # Convert to tensors
            states = torch.tensor([e.state for e in experiences], device=self.device)
            actions = torch.tensor([e.action for e in experiences], device=self.device)
            rewards = torch.tensor([e.reward for e in experiences], device=self.device)
            next_states = torch.tensor([e.next_state for e in experiences], device=self.device)
            dones = torch.tensor([e.done for e in experiences], device=self.device)
            weights = torch.tensor(weights, device=self.device)
            
            return {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_states,
                'dones': dones,
                'weights': weights,
                'indices': indices
            }
        
        else:
            # Standard sampling
            experiences = self.buffer.sample(batch_size)
            
            if isinstance(experiences, tuple):
                experiences, source_counts = experiences
                logger.debug(f"Sampled from sources: {source_counts}")
            
            states = torch.tensor([e.state for e in experiences], device=self.device)
            actions = torch.tensor([e.action for e in experiences], device=self.device)
            rewards = torch.tensor([e.reward for e in experiences], device=self.device)
            next_states = torch.tensor([e.next_state for e in experiences], device=self.device)
            dones = torch.tensor([e.done for e in experiences], device=self.device)
            
            return {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_states,
                'dones': dones,
                'weights': torch.ones(batch_size, device=self.device)
            }
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities if using prioritized replay."""
        if hasattr(self.buffer, 'update_priorities'):
            self.buffer.update_priorities(indices, td_errors)


# ==================== FILE 2: ppo_tuner_bubble.py ====================

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional
import torch
import numpy as np
from datetime import datetime
from bubbles_core import UniversalBubble, Actions, Event, UniversalCode, Tags, SystemContext, logger, EventService
from bayes_opt import BayesianOptimization


class PPOConfig:
    """Configuration for PPO hyperparameters."""
    
    def __init__(self, **kwargs):
        # Core PPO parameters
        self.gamma = kwargs.get('gamma', 0.99)
        self.lam = kwargs.get('lam', 0.95)
        self.clip_eps = kwargs.get('clip_eps', 0.2)
        self.vf_coef = kwargs.get('vf_coef', 0.5)
        self.ent_coef = kwargs.get('ent_coef', 0.01)
        self.ppo_epochs = kwargs.get('ppo_epochs', 10)
        self.batch_size = kwargs.get('batch_size', 64)
        
        # Learning parameters
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.gradient_clip_norm = kwargs.get('gradient_clip_norm', 0.5)
        
        # Architecture parameters
        self.hidden_layers = kwargs.get('hidden_layers', 2)
        self.hidden_dim = kwargs.get('hidden_dim', 512)
        self.dropout_rate = kwargs.get('dropout_rate', 0.1)
        
        # Advanced features
        self.spawn_threshold = kwargs.get('spawn_threshold', 0.99)
        self.max_algorithms = kwargs.get('max_algorithms', 10)
        self.weight_performance = kwargs.get('weight_performance', 0.3)
        self.weight_stability = kwargs.get('weight_stability', 0.3)
        self.weight_efficiency = kwargs.get('weight_efficiency', 0.2)
        self.weight_innovation = kwargs.get('weight_innovation', 0.2)
        
        # Buffer settings
        self.experience_buffer_size = kwargs.get('experience_buffer_size', 10000)
        self.cache_size = kwargs.get('cache_size', 1000)
        self.cache_ttl = kwargs.get('cache_ttl', 300)
        self.pool_size = kwargs.get('pool_size', 3)


class PPOHyperparameterTuner:
    """Hyperparameter tuner specifically for PPO."""
    
    def __init__(self, context: SystemContext, num_iterations: int = 50, 
                 num_episodes: int = 10, num_trials: int = 30,
                 parallel_workers: int = 1):
        self.context = context
        self.num_iterations = num_iterations
        self.num_episodes = num_episodes
        self.num_trials = num_trials
        self.parallel_workers = parallel_workers
        self.results = []
        self.plot_dir = f'ppo_tuning_plots_{int(datetime.now().timestamp())}'
        os.makedirs(self.plot_dir, exist_ok=True)
        
        self.best_config = None
        self.best_score = float('-inf')
        
        logger.info(f"PPOHyperparameterTuner: Initialized for PPO optimization")
    
    async def evaluate_config(self, config: Dict[str, float]) -> Dict[str, float]:
        """Evaluate a PPO hyperparameter configuration."""
        try:
            # Import PPO here to avoid circular imports
            from ppo_bubble_fixed import FullEnhancedPPOWithMetaLearning
            
            # Create PPO configuration
            config_obj = PPOConfig(
                gamma=config['gamma'],
                lam=config['lam'],
                clip_eps=config['clip_eps'],
                vf_coef=config['vf_coef'],
                ent_coef=config['ent_coef'],
                ppo_epochs=int(config['ppo_epochs']),
                batch_size=int(config['batch_size']),
                learning_rate=config['learning_rate'],
                gradient_clip_norm=config['gradient_clip_norm'],
                hidden_layers=int(config['hidden_layers']),
                hidden_dim=int(config['hidden_dim']),
                dropout_rate=config['dropout_rate'],
                spawn_threshold=config['spawn_threshold'],
                weight_performance=config['weight_performance'],
                weight_stability=config['weight_stability']
            )
            
            # Create unique ID
            config_id = f"ppo_lr{config['learning_rate']:.1e}_clip{config['clip_eps']:.2f}_bs{config_obj.batch_size}"
            
            # Create PPO instance
            ppo = FullEnhancedPPOWithMetaLearning(
                object_id=f"ppo_tune_{config_id}",
                context=self.context,
                state_dim=32,
                action_dim=16,
                **config_obj.__dict__
            )
            
            # Wait for initialization
            await asyncio.sleep(1)
            
            # Run training episodes
            episode_rewards = []
            stability_violations = 0
            algorithm_spawns = 0
            
            for episode in range(self.num_episodes):
                # Reset environment
                if ppo.env:
                    state = ppo.env.reset()
                    episode_reward = 0
                    episode_length = 0
                    
                    # Run episode
                    for step in range(self.num_iterations):
                        # Get action
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(ppo.device)
                        with torch.no_grad():
                            action, _, value = ppo.get_action_and_value(state_tensor)
                        
                        # Step environment
                        next_state, reward, done = await ppo.env.step(action.squeeze(0))
                        
                        episode_reward += reward
                        episode_length += 1
                        
                        # Check for instability
                        if abs(reward) > 10:
                            stability_violations += 1
                        
                        state = next_state
                        
                        if done:
                            break
                    
                    episode_rewards.append(episode_reward)
                    
                    # Check spawned algorithms
                    algorithm_spawns = len(ppo.spawned_algorithms)
            
            # Calculate metrics
            result = {
                'config': config,
                'avg_reward': np.mean(episode_rewards) if episode_rewards else -1000.0,
                'std_reward': np.std(episode_rewards) if episode_rewards else 0.0,
                'stability_violations': stability_violations,
                'algorithms_spawned': algorithm_spawns,
                'cache_hit_rate': ppo.prediction_cache.get_hit_rate(),
                'pool_hit_rate': ppo.bubble_pool.get_hit_rate(),
                'final_stage': ppo.curriculum.current_stage,
                'patterns_discovered': ppo.patterns_discovered
            }
            
            # Calculate composite score
            score = (
                result['avg_reward'] * 0.4 +
                (1.0 - min(1.0, stability_violations / 100)) * 0.3 +
                min(1.0, algorithm_spawns / 5) * 0.2 +
                result['cache_hit_rate'] * 0.1
            )
            
            result['composite_score'] = score
            
            logger.info(f"Evaluated PPO config {config_id}: "
                       f"avg_reward={result['avg_reward']:.4f}, "
                       f"stability={stability_violations}, "
                       f"score={score:.4f}")
            
            # Clean up
            del ppo
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating PPO config {config}: {e}", exc_info=True)
            return {
                'config': config,
                'avg_reward': -1000.0,
                'std_reward': 0.0,
                'stability_violations': 1000,
                'algorithms_spawned': 0,
                'cache_hit_rate': 0.0,
                'pool_hit_rate': 0.0,
                'final_stage': 0,
                'patterns_discovered': 0,
                'composite_score': -1000.0
            }
    
    async def run_bayesian_search(self) -> Optional[Dict]:
        """Run Bayesian optimization to find optimal hyperparameters."""
        try:
            # Store the current event loop for thread communication
            current_loop = asyncio.get_running_loop()
            
            async def objective(**params):
                """Async objective function for Bayesian optimization."""
                result = await self.evaluate_config(params)
                self.results.append(result)
                return result['composite_score']
            
            def sync_objective(**params):
                """Synchronous wrapper for the async objective function."""
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    objective(**params), 
                    current_loop
                )
                
                try:
                    return future.result(timeout=600)  # 10 minute timeout
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout evaluating config: {params}")
                    return -1000.0
                except Exception as e:
                    logger.error(f"Error in sync_objective: {e}", exc_info=True)
                    return -1000.0
            
            # Define parameter bounds for PPO
            pbounds = {
                'gamma': (0.9, 0.999),
                'lam': (0.9, 0.99),
                'clip_eps': (0.1, 0.3),
                'vf_coef': (0.1, 1.0),
                'ent_coef': (0.0001, 0.1),
                'ppo_epochs': (5, 20),
                'batch_size': (32, 256),
                'learning_rate': (1e-5, 1e-3),
                'gradient_clip_norm': (0.1, 1.0),
                'hidden_layers': (2, 4),
                'hidden_dim': (256, 1024),
                'dropout_rate': (0.0, 0.3),
                'spawn_threshold': (0.9, 0.999),
                'weight_performance': (0.1, 0.5),
                'weight_stability': (0.1, 0.5)
            }
            
            # Run optimization in a thread pool
            import concurrent.futures
            def run_optimization():
                """Run the Bayesian optimization in a separate thread."""
                optimizer = BayesianOptimization(
                    f=sync_objective,
                    pbounds=pbounds,
                    random_state=42,
                    verbose=2,
                    allow_duplicate_points=True
                )
                
                # Run optimization
                optimizer.maximize(
                    init_points=min(5, self.num_trials // 3),
                    n_iter=self.num_trials - min(5, self.num_trials // 3)
                )
                
                return optimizer.max
            
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                best_result = await loop.run_in_executor(executor, run_optimization)
            
            logger.info(f"Completed {self.num_trials} tuning trials. Best result: {best_result}")
            
            # Find best configuration from results
            if self.results:
                best_config_result = max(self.results, key=lambda x: x['composite_score'])
                
                logger.info(f"Best PPO configuration found: {best_config_result['config']}")
                logger.info(f"Best score: {best_config_result['composite_score']:.4f}")
                
                return best_config_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error in Bayesian search: {e}", exc_info=True)
            raise


class PPOTuningBubble(UniversalBubble):
    """Bubble for tuning PPO hyperparameters."""
    
    def __init__(self, object_id: str, context: SystemContext, ppo_bubble_id: str,
                 num_iterations: int = 50, num_episodes: int = 10, num_trials: int = 30,
                 cpu_threshold: float = 80.0, memory_threshold: float = 85.0):
        super().__init__(object_id=object_id, context=context)
        
        self.ppo_bubble_id = ppo_bubble_id
        self.tuner = PPOHyperparameterTuner(
            self.context,
            num_iterations=num_iterations,
            num_episodes=num_episodes,
            num_trials=num_trials
        )
        
        self.best_config: Optional[PPOConfig] = None
        self.best_metrics: Optional[Dict] = None
        self.tuning_active = False
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        
        # Dynamic adjustment parameters
        self.high_load_adjustments = {
            'clip_eps_multiplier': 1.2,
            'batch_size_divisor': 2,
            'learning_rate_multiplier': 0.8
        }
        
        # Initialize async tasks
        self._initialized = False
        self._init_task = asyncio.create_task(self._async_init())
        
        logger.info(f"{self.object_id}: Initialized PPOTuningBubble for '{ppo_bubble_id}'")
    
    async def _async_init(self):
        """Perform async initialization."""
        try:
            await self._subscribe_to_events()
            self._initialized = True
        except Exception as e:
            logger.error(f"{self.object_id}: Failed async initialization: {e}", exc_info=True)
            raise
    
    async def ensure_initialized(self):
        """Ensure async initialization is complete."""
        if not self._initialized:
            await self._init_task
    
    async def _subscribe_to_events(self):
        """Subscribe to system events."""
        await asyncio.sleep(0.1)
        try:
            await EventService.subscribe(Actions.SYSTEM_STATE_UPDATE, self.handle_event)
            logger.debug(f"{self.object_id}: Subscribed to SYSTEM_STATE_UPDATE")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to events: {e}", exc_info=True)
    
    async def handle_event(self, event: Event):
        """Handle events to adjust tuning based on system state."""
        if event.type == Actions.SYSTEM_STATE_UPDATE and self.best_config:
            if isinstance(event.data, UniversalCode) and event.data.tag == Tags.DICT:
                state = event.data.value
                cpu_usage = state.get("cpu_percent", 0)
                memory_usage = state.get("memory_percent", 0)
                
                # Dynamic adjustments
                from ppo_bubble_fixed import FullEnhancedPPOWithMetaLearning
                ppo = self.context.get_bubble(self.ppo_bubble_id)
                if ppo and isinstance(ppo, FullEnhancedPPOWithMetaLearning):
                    adjustments_made = []
                    
                    # High CPU adjustment
                    if cpu_usage > self.cpu_threshold:
                        new_clip = min(0.5, ppo.clip_eps * self.high_load_adjustments['clip_eps_multiplier'])
                        if new_clip != ppo.clip_eps:
                            ppo.clip_eps = new_clip
                            adjustments_made.append(f"clip_eps→{new_clip:.2f}")
                        
                        # Reduce learning rate
                        for param_group in ppo.optimizer.param_groups:
                            new_lr = param_group['lr'] * self.high_load_adjustments['learning_rate_multiplier']
                            param_group['lr'] = new_lr
                            adjustments_made.append(f"lr→{new_lr:.1e}")
                    
                    # High memory adjustment
                    if memory_usage > self.memory_threshold:
                        new_batch_size = max(16, ppo.batch_size // self.high_load_adjustments['batch_size_divisor'])
                        if new_batch_size != ppo.batch_size:
                            ppo.batch_size = new_batch_size
                            adjustments_made.append(f"batch_size→{new_batch_size}")
                    
                    if adjustments_made:
                        logger.info(f"{self.object_id}: Dynamic PPO adjustments "
                                   f"(CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%): "
                                   f"{', '.join(adjustments_made)}")
        
        await super().handle_event(event)
    
    def _create_config_from_dict(self, config_dict: Dict) -> PPOConfig:
        """Create PPOConfig from dictionary."""
        return PPOConfig(**config_dict)
    
    async def run_tuning(self):
        """Run Bayesian optimization for PPO hyperparameters."""
        if self.tuning_active:
            logger.warning(f"{self.object_id}: Tuning already in progress.")
            return
        
        await self.ensure_initialized()
        self.tuning_active = True
        
        try:
            logger.info(f"{self.object_id}: Starting PPO hyperparameter tuning")
            
            # Run Bayesian optimization
            result = await self.tuner.run_bayesian_search()
            
            if result:
                # Create config object from best parameters
                self.best_config = self._create_config_from_dict(result['config'])
                self.best_metrics = {
                    'avg_reward': result['avg_reward'],
                    'stability_violations': result['stability_violations'],
                    'algorithms_spawned': result['algorithms_spawned'],
                    'composite_score': result['composite_score']
                }
                
                logger.info(f"{self.object_id}: Best PPO config found - "
                           f"gamma: {self.best_config.gamma:.3f}, "
                           f"clip_eps: {self.best_config.clip_eps:.2f}, "
                           f"lr: {self.best_config.learning_rate:.1e}, "
                           f"score: {self.best_metrics['composite_score']:.3f}")
                
                # Apply best configuration
                from ppo_bubble_fixed import FullEnhancedPPOWithMetaLearning
                ppo = self.context.get_bubble(self.ppo_bubble_id)
                if ppo and isinstance(ppo, FullEnhancedPPOWithMetaLearning):
                    # Update PPO parameters
                    ppo.gamma = self.best_config.gamma
                    ppo.lam = self.best_config.lam
                    ppo.clip_eps = self.best_config.clip_eps
                    ppo.vf_coef = self.best_config.vf_coef
                    ppo.ent_coef = self.best_config.ent_coef
                    ppo.ppo_epochs = self.best_config.ppo_epochs
                    ppo.batch_size = self.best_config.batch_size
                    
                    # Update optimizer
                    for param_group in ppo.optimizer.param_groups:
                        param_group['lr'] = self.best_config.learning_rate
                    
                    logger.info(f"{self.object_id}: Updated PPO '{self.ppo_bubble_id}' with best config")
                
                # Cache results
                if self.context.response_cache:
                    cache_key = f"ppo_tuning_{self.ppo_bubble_id}"
                    cache_data = {
                        "best_config": result['config'],
                        **self.best_metrics
                    }
                    await self.context.response_cache.put(cache_key, json.dumps(cache_data))
                    logger.info(f"{self.object_id}: Cached PPO tuning results")
                
                # Notify completion
                await self.add_chat_message(
                    f"PPO tuning completed for {self.ppo_bubble_id}. "
                    f"Best score: {self.best_metrics['composite_score']:.3f}, "
                    f"Avg reward: {self.best_metrics['avg_reward']:.2f}"
                )
            else:
                logger.warning(f"{self.object_id}: No valid tuning results found")
                await self.add_chat_message(f"PPO tuning failed: No valid results")
                
        except Exception as e:
            logger.error(f"{self.object_id}: Error in PPO tuning: {e}", exc_info=True)
            await self.add_chat_message(f"PPO tuning failed: {str(e)}")
        finally:
            self.tuning_active = False
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def load_cached_results(self) -> bool:
        """Load cached tuning results if available."""
        if not self.context.response_cache:
            return False
        
        try:
            cache_key = f"ppo_tuning_{self.ppo_bubble_id}"
            cached_results = await self.context.response_cache.get(cache_key)
            
            if cached_results:
                tuning_data = json.loads(cached_results)
                
                # Extract config and metrics
                config_dict = tuning_data.get("best_config")
                if config_dict:
                    self.best_config = self._create_config_from_dict(config_dict)
                    self.best_metrics = {
                        k: v for k, v in tuning_data.items() 
                        if k != "best_config"
                    }
                    
                    logger.info(f"{self.object_id}: Loaded cached PPO tuning results")
                    
                    # Apply cached configuration
                    from ppo_bubble_fixed import FullEnhancedPPOWithMetaLearning
                    ppo = self.context.get_bubble(self.ppo_bubble_id)
                    if ppo and isinstance(ppo, FullEnhancedPPOWithMetaLearning):
                        ppo.gamma = self.best_config.gamma
                        ppo.lam = self.best_config.lam
                        ppo.clip_eps = self.best_config.clip_eps
                        ppo.vf_coef = self.best_config.vf_coef
                        ppo.ent_coef = self.best_config.ent_coef
                        ppo.ppo_epochs = self.best_config.ppo_epochs
                        ppo.batch_size = self.best_config.batch_size
                        
                        logger.info(f"{self.object_id}: Applied cached config to PPO")
                        return True
                        
        except Exception as e:
            logger.error(f"{self.object_id}: Error loading cached results: {e}", exc_info=True)
        
        return False
    
    async def autonomous_step(self):
        """Periodically run tuning or check cache."""
        await super().autonomous_step()
        
        # Run tuning every 600 steps
        if self.execution_count % 600 == 0:
            # First try to load cached results
            cached_loaded = await self.load_cached_results()
            
            if not cached_loaded and not self.tuning_active:
                # No cached results, run tuning
                logger.info(f"{self.object_id}: Starting scheduled PPO tuning")
                await self.run_tuning()
        
        # Periodic memory cleanup
        if self.execution_count % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        await asyncio.sleep(0.5)


# ==================== FILE 3: ppo_with_replay_integration.py ====================

import torch
import numpy as np
from typing import Dict, Optional, List
import asyncio
import logging

# Import the replay buffers we just defined
# from advanced_replay_buffers import ReplayBufferWrapper, PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


class PPOWithAdvancedReplay:
    """
    Enhanced PPO with advanced replay buffer capabilities
    This is a mixin/extension for your existing PPO class
    """
    
    def __init__(self, *args, replay_buffer_type: str = "prioritized",
                 replay_capacity: int = 10000, use_replay: bool = True,
                 replay_ratio: float = 0.5, **kwargs):
        
        # Initialize parent PPO
        super().__init__(*args, **kwargs)
        
        self.use_replay = use_replay
        self.replay_ratio = replay_ratio
        
        if self.use_replay:
            # Create replay buffer
            buffer_kwargs = {
                'alpha': kwargs.get('replay_alpha', 0.6),
                'beta': kwargs.get('replay_beta', 0.4),
                'n_steps': kwargs.get('n_steps', 3),
                'k': kwargs.get('her_k', 4),
                'rare_threshold': kwargs.get('rare_threshold', 0.1)
            }
            
            self.replay_buffer = ReplayBufferWrapper(
                buffer_type=replay_buffer_type,
                capacity=replay_capacity,
                device=str(self.device),
                **buffer_kwargs
            )
            
            logger.info(f"{self.object_id}: Initialized with {replay_buffer_type} replay buffer")
        else:
            self.replay_buffer = None
    
    async def train_with_replay(self):
        """Enhanced training loop with replay buffer integration."""
        if not self.env:
            logger.warning(f"{self.object_id}: Cannot train without environment")
            return
        
        logger.info(f"{self.object_id}: Starting training with replay buffer")
        
        episodes = 0
        while not self.context.stop_event.is_set():
            try:
                # Standard PPO rollout collection
                complexity = self.curriculum.get_current_complexity()
                state = self.env.reset(complexity)
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                episode_reward = 0
                episode_length = 0
                
                # Rollout buffers
                states = []
                actions = []
                log_probs = []
                values = []
                rewards = []
                dones = []
                next_states = []
                
                for step in range(self.env.max_steps):
                    # Get action and value
                    with torch.no_grad():
                        action, log_prob, value = self.get_action_and_value(state)
                    
                    # Store rollout data
                    states.append(state.squeeze(0))
                    actions.append(action.squeeze(0))
                    log_probs.append(log_prob)
                    values.append(value)
                    
                    # Execute action
                    next_state, reward, done = await self.env.step(action.squeeze(0))
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                    
                    rewards.append(reward)
                    dones.append(done)
                    next_states.append(next_state)
                    
                    # Store in replay buffer if enabled
                    if self.use_replay and self.replay_buffer:
                        self.replay_buffer.buffer.push(
                            state.squeeze(0).cpu().numpy(),
                            action.squeeze(0).cpu().numpy(),
                            reward,
                            next_state.cpu().numpy(),
                            done,
                            {'episode': episodes, 'step': step}
                        )
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    state = next_state.unsqueeze(0)
                    
                    if done:
                        break
                
                # Stack on-policy data
                states = torch.stack(states)
                actions = torch.stack(actions)
                log_probs = torch.stack(log_probs)
                values = torch.stack(values)
                rewards = torch.tensor(rewards, device=self.device)
                dones = torch.tensor(dones, device=self.device)
                
                # Get last value for GAE
                with torch.no_grad():
                    _, _, next_value = self.get_action_and_value(state)
                
                # Compute advantages and returns using GAE
                advantages, returns = self.compute_gae(rewards, values, next_value.item(), dones)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Combine on-policy and replay data if enabled
                if self.use_replay and len(self.replay_buffer.buffer) > self.batch_size:
                    await self._train_with_mixed_data(
                        states, actions, log_probs, returns, advantages
                    )
                else:
                    # Standard PPO update
                    self.ppo_update(states, actions, log_probs, returns, advantages)
                
                episodes += 1
                self.decisions_made += episode_length
                
                # Log progress
                if episodes % 10 == 0:
                    logger.info(f"{self.object_id}: Episode {episodes} - "
                              f"Reward: {episode_reward:.2f}, "
                              f"Replay buffer size: {len(self.replay_buffer.buffer) if self.replay_buffer else 0}")
                    
                    self.save_model()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"{self.object_id}: Training error: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _train_with_mixed_data(self, on_policy_states: torch.Tensor,
                                    on_policy_actions: torch.Tensor,
                                    on_policy_log_probs: torch.Tensor,
                                    on_policy_returns: torch.Tensor,
                                    on_policy_advantages: torch.Tensor):
        """Training with mixed on-policy and replay data."""
        
        # Determine batch sizes
        total_batch_size = self.batch_size
        replay_batch_size = int(total_batch_size * self.replay_ratio)
        on_policy_batch_size = total_batch_size - replay_batch_size
        
        for epoch in range(self.ppo_epochs):
            # Sample from on-policy data
            on_policy_indices = torch.randperm(len(on_policy_states))[:on_policy_batch_size]
            
            batch_states = on_policy_states[on_policy_indices]
            batch_actions = on_policy_actions[on_policy_indices]
            batch_old_log_probs = on_policy_log_probs[on_policy_indices]
            batch_returns = on_policy_returns[on_policy_indices]
            batch_advantages = on_policy_advantages[on_policy_indices]
            
            # Sample from replay buffer
            if replay_batch_size > 0:
                replay_data = self.replay_buffer.sample_for_ppo(replay_batch_size)
                
                # Compute returns and advantages for replay data
                replay_values = self.value_net(replay_data['states']).squeeze(-1)
                replay_next_values = self.value_net(replay_data['next_states']).squeeze(-1)
                
                # Simple 1-step TD for replay
                replay_returns = replay_data['rewards'] + self.gamma * replay_next_values * (1 - replay_data['dones'])
                replay_advantages = replay_returns - replay_values
                
                # Normalize replay advantages
                replay_advantages = (replay_advantages - replay_advantages.mean()) / (replay_advantages.std() + 1e-8)
                
                # Combine batches
                combined_states = torch.cat([batch_states, replay_data['states']])
                combined_actions = torch.cat([batch_actions, replay_data['actions']])
                combined_returns = torch.cat([batch_returns, replay_returns])
                combined_advantages = torch.cat([batch_advantages, replay_advantages])
                
                # Get old log probs for replay data
                with torch.no_grad():
                    replay_dist = self._get_distribution(replay_data['states'])
                    replay_old_log_probs = replay_dist.log_prob(replay_data['actions']).sum(dim=-1)
                
                combined_old_log_probs = torch.cat([batch_old_log_probs, replay_old_log_probs])
                
                # Use importance weights if available
                if 'weights' in replay_data:
                    on_policy_weights = torch.ones(on_policy_batch_size, device=self.device)
                    combined_weights = torch.cat([on_policy_weights, replay_data['weights']])
                else:
                    combined_weights = torch.ones(len(combined_states), device=self.device)
                
                # Perform weighted PPO update
                self._weighted_ppo_step(
                    combined_states, combined_actions, combined_old_log_probs,
                    combined_returns, combined_advantages, combined_weights
                )
                
                # Update priorities if using prioritized replay
                if 'indices' in replay_data and hasattr(self.replay_buffer.buffer, 'update_priorities'):
                    # Calculate new TD errors
                    with torch.no_grad():
                        new_values = self.value_net(replay_data['states']).squeeze(-1)
                        td_errors = replay_returns - new_values
                    
                    self.replay_buffer.update_priorities(
                        replay_data['indices'],
                        td_errors.cpu().numpy()
                    )
            else:
                # Just use on-policy data
                self.ppo_update(batch_states, batch_actions, batch_old_log_probs, 
                              batch_returns, batch_advantages)
    
    def _weighted_ppo_step(self, states: torch.Tensor, actions: torch.Tensor,
                          old_log_probs: torch.Tensor, returns: torch.Tensor,
                          advantages: torch.Tensor, weights: torch.Tensor):
        """Single PPO optimization step with importance weights."""
        
        # Get new log_probs and values
        _, new_log_probs, new_values = self.get_action_and_value(states)
        
        # Policy loss with clipped surrogate objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        
        # Apply importance weights
        policy_loss = -(torch.min(surr1, surr2) * weights).mean()
        
        # Value loss with importance weights
        value_loss = (weights * (new_values - returns) ** 2).mean()
        
        # Entropy for exploration
        dist = self._get_distribution(states)
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.all_parameters, 0.5)
        self.optimizer.step()


# ==================== FILE 4: complete_ppo_system.py ====================

"""
Complete PPO System Integration
Combines everything into a ready-to-use system
"""

import asyncio
from typing import Dict, Optional
from bubbles_core import SystemContext
from ppo_bubble_fixed import FullEnhancedPPOWithMetaLearning

# Create enhanced PPO class with replay support
class CompleteEnhancedPPO(PPOWithAdvancedReplay, FullEnhancedPPOWithMetaLearning):
    """Complete PPO with all enhancements."""
    
    def __init__(self, object_id: str, context: SystemContext, **kwargs):
        # Initialize with replay support
        super().__init__(
            object_id=object_id,
            context=context,
            **kwargs
        )
        
        logger.info(f"{self.object_id}: Complete Enhanced PPO initialized with all features")
    
    # Override train method to use replay-enhanced training
    async def train(self):
        """Use replay-enhanced training if enabled."""
        if self.use_replay:
            await self.train_with_replay()
        else:
            await super().train()  # Use parent's train method


async def create_ppo_system(context: SystemContext, config: Optional[Dict] = None) -> Dict:
    """
    Create a complete PPO system with all features.
    
    Args:
        context: SystemContext for bubble operations
        config: Optional configuration dictionary
    
    Returns:
        Dictionary with created PPO and tuner instances
    """
    
    # Default configuration
    default_config = {
        # Core dimensions
        'state_dim': 32,
        'action_dim': 16,
        
        # Replay buffer settings
        'replay_buffer_type': 'prioritized',  # Options: 'prioritized', 'her', 'nstep', 'combined'
        'replay_capacity': 10000,
        'use_replay': True,
        'replay_ratio': 0.3,  # 30% replay data
        'replay_alpha': 0.6,
        'replay_beta': 0.4,
        
        # PPO hyperparameters (will be tuned)
        'gamma': 0.99,
        'lam': 0.95,
        'clip_eps': 0.2,
        'vf_coef': 0.5,
        'ent_coef': 0.01,
        'ppo_epochs': 10,
        'batch_size': 64,
        
        # Learning parameters
        'learning_rate': 3e-4,
        'gradient_clip_norm': 0.5,
        
        # Advanced features
        'spawn_threshold': 0.95,
        'max_algorithms': 10,
        'experience_buffer_size': 10000,
        'cache_size': 1000,
        'cache_ttl': 300,
        'pool_size': 3,
        
        # Tuning settings
        'tune_on_start': True,
        'tuning_trials': 30,
        'tuning_episodes': 10,
        'tuning_iterations': 50
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)
    
    # Create main PPO instance
    ppo = CompleteEnhancedPPO(
        object_id="enhanced_ppo_main",
        context=context,
        **default_config
    )
    
    # Create tuning bubble
    tuner = PPOTuningBubble(
        object_id="ppo_tuner",
        context=context,
        ppo_bubble_id="enhanced_ppo_main",
        num_iterations=default_config['tuning_iterations'],
        num_episodes=default_config['tuning_episodes'],
        num_trials=default_config['tuning_trials']
    )
    
    # Run initial tuning if requested
    if default_config['tune_on_start']:
        logger.info("Starting initial hyperparameter tuning...")
        await tuner.run_tuning()
    
    return {
        'ppo': ppo,
        'tuner': tuner,
        'config': default_config
    }


# ==================== USAGE EXAMPLES ====================

async def example_basic_usage(context: SystemContext):
    """Basic usage example."""
    
    # Create PPO system with default settings
    system = await create_ppo_system(context)
    
    ppo = system['ppo']
    tuner = system['tuner']
    
    # The PPO is now training automatically with tuned hyperparameters
    # and advanced replay buffer
    
    # Check status after some time
    await asyncio.sleep(60)
    
    status = await ppo.get_status()
    print(f"PPO Status: {status}")


async def example_custom_replay_buffer(context: SystemContext):
    """Example with custom replay buffer configuration."""
    
    # Configure for N-step returns with combined buffer
    config = {
        'replay_buffer_type': 'combined',
        'replay_capacity': 20000,
        'replay_ratio': 0.5,  # 50% replay data
        'rare_threshold': 0.05,  # 5% chance for rare event classification
        'n_steps': 5,  # For n-step returns
        'buffer_ratios': {
            'uniform': 0.4,
            'prioritized': 0.4,
            'rare': 0.2
        }
    }
    
    system = await create_ppo_system(context, config)
    
    # Access the PPO
    ppo = system['ppo']
    
    # The PPO is now using a combined replay buffer with
    # uniform sampling, prioritized experience replay, and rare event replay


async def example_goal_conditioned_ppo(context: SystemContext):
    """Example of goal-conditioned PPO with HER."""
    
    # Configure for HER
    config = {
        'replay_buffer_type': 'her',
        'replay_capacity': 50000,
        'use_replay': True,
        'replay_ratio': 0.8,  # High replay ratio for HER
        'her_k': 4,  # Number of hindsight goals per transition
        'state_dim': 40,  # Includes goal dimensions
        'action_dim': 8
    }
    
    system = await create_ppo_system(context, config)
    
    ppo = system['ppo']
    
    # For HER, you need to structure your environment to provide goals
    # The replay buffer will automatically generate hindsight experiences


async def example_manual_tuning(context: SystemContext):
    """Example of manual hyperparameter tuning."""
    
    # Create system without auto-tuning
    config = {'tune_on_start': False}
    system = await create_ppo_system(context, config)
    
    ppo = system['ppo']
    tuner = system['tuner']
    
    # Let it train with default parameters
    await asyncio.sleep(300)  # 5 minutes
    
    # Check performance
    status1 = await ppo.get_status()
    print(f"Performance before tuning: {status1['recent_reward_avg']}")
    
    # Run tuning
    await tuner.run_tuning()
    
    # Let it train with tuned parameters
    await asyncio.sleep(300)  # 5 minutes
    
    # Check improved performance
    status2 = await ppo.get_status()
    print(f"Performance after tuning: {status2['recent_reward_avg']}")


async def example_monitoring(context: SystemContext):
    """Example of monitoring PPO performance."""
    
    system = await create_ppo_system(context)
    ppo = system['ppo']
    
    # Monitor for 10 minutes
    for i in range(20):
        await asyncio.sleep(30)  # Check every 30 seconds
        
        status = await ppo.get_status()
        
        print(f"\n=== PPO Status at {i*30}s ===")
        print(f"Episodes: {status.get('decisions_made', 0) // 100}")
        print(f"Avg Reward: {status.get('recent_reward_avg', 0):.3f}")
        print(f"Curriculum Stage: {status.get('curriculum_stage', 'unknown')}")
        print(f"Algorithms Spawned: {status.get('algorithms_spawned', 0)}")
        print(f"Patterns Discovered: {status.get('patterns_discovered', 0)}")
        
        if 'replay_buffer' in status:
            rb_stats = status['replay_buffer']
            print(f"Replay Buffer Size: {rb_stats.get('replay_buffer_size', 0)}")
            print(f"Replay Buffer Type: {rb_stats.get('replay_buffer_type', 'none')}")


# ==================== MAIN ENTRY POINT ====================

async def main():
    """Main entry point for testing."""
    
    # Create a mock context for testing
    # In production, you'll use your actual SystemContext
    from bubbles_core import SystemContext
    
    context = SystemContext()
    
    # Run basic example
    await example_basic_usage(context)


if __name__ == "__main__":
    asyncio.run(main())
