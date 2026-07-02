"""
differentiable_planner.py - Differentiable Planner
Finds optimal action sequences by backpropagating through the world model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class DifferentiablePlanner(nn.Module):
    """
    Differentiable planner that uses gradient descent to find optimal action sequences
    """
    
    def __init__(self, world_model: nn.Module, horizon: int = 10, 
                 action_dim: int = 2, action_scale: float = 1.5):
        super().__init__()
        self.world_model = world_model
        self.horizon = horizon
        self.action_dim = action_dim
        self.action_scale = action_scale
    
    def plan(self, initial_latent: torch.Tensor, target_state: torch.Tensor,
             num_iterations: int = 100, lr: float = 0.01) -> torch.Tensor:
        """
        Plan optimal action sequence using gradient descent
        
        initial_latent: [batch, latent_dim] - current world state
        target_state: [batch, latent_dim] - desired state
        Returns: [batch, horizon, action_dim] - optimal action sequence
        """
        batch_size = initial_latent.shape[0]
        
        # Initialize action sequence as learnable parameter
        actions = torch.randn(batch_size, self.horizon, self.action_dim, 
                              requires_grad=True, device=initial_latent.device)
        actions = actions * self.action_scale * 0.1  # Small initial values
        
        # Use Adam optimizer
        optimizer = torch.optim.Adam([actions], lr=lr)
        
        best_actions = actions.clone().detach()
        best_loss = float('inf')
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Simulate trajectory
            trajectory = self.world_model.simulate_rollout(initial_latent, actions)
            
            # Final state
            final_latent = trajectory[:, -1, :]
            
            # Loss: distance to target + action smoothness penalty
            target_loss = F.mse_loss(final_latent, target_state)
            
            # Action smoothness penalty (encourage smooth trajectories)
            smoothness_loss = torch.mean((actions[:, 1:, :] - actions[:, :-1, :]) ** 2)
            
            # Action magnitude penalty
            magnitude_loss = torch.mean(actions ** 2) * 0.01
            
            loss = target_loss + 0.1 * smoothness_loss + magnitude_loss
            
            # Backpropagate
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([actions], max_norm=1.0)
            
            optimizer.step()
            
            # Clamp actions
            actions.data = torch.clamp(actions.data, -self.action_scale, self.action_scale)
            
            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_actions = actions.clone().detach()
            
            if iteration % 20 == 0:
                print(f"  Iteration {iteration}: loss = {loss.item():.6f}")
        
        return best_actions
    
    def plan_mppi(self, initial_latent: torch.Tensor, target_state: torch.Tensor,
                  num_samples: int = 100, temperature: float = 0.1) -> torch.Tensor:
        """
        MPPI (Model Predictive Path Integral) planning
        Samples random actions, weights them by cost, returns weighted average
        
        This is a gradient-free alternative to the differentiable planner
        """
        batch_size = initial_latent.shape[0]
        device = initial_latent.device
        
        # Sample random action sequences
        samples = torch.randn(num_samples, batch_size, self.horizon, self.action_dim, device=device)
        samples = samples * self.action_scale * 0.5
        
        # Compute costs for each sample
        costs = []
        
        for i in range(num_samples):
            actions_i = samples[i]
            trajectory = self.world_model.simulate_rollout(initial_latent, actions_i)
            final_latent = trajectory[:, -1, :]
            
            cost = F.mse_loss(final_latent, target_state, reduction='none')
            cost = cost.mean(dim=1).sum()
            
            # Smoothness penalty
            smooth_cost = torch.mean((actions_i[:, 1:, :] - actions_i[:, :-1, :]) ** 2)
            cost = cost + 0.1 * smooth_cost
            
            costs.append(cost)
        
        costs = torch.stack(costs)  # [num_samples]
        
        # Weighted average (lower cost = higher weight)
        weights = F.softmax(-costs / temperature, dim=0)
        
        # Compute weighted average action
        weighted_actions = torch.zeros(batch_size, self.horizon, self.action_dim, device=device)
        for i in range(num_samples):
            weighted_actions += weights[i] * samples[i]
        
        return weighted_actions


class PlanningController:
    """
    High-level controller that combines planning and execution
    """
    
    def __init__(self, world_model: nn.Module, planner: DifferentiablePlanner,
                 latent_dim: int = 64):
        self.world_model = world_model
        self.planner = planner
        self.latent_dim = latent_dim
        self.current_latent = None
        self.current_object_ids = []
    
    def reset(self, object_embeddings: torch.Tensor):
        """Initialize controller with initial scene"""
        self.current_latent = self.world_model.encode_objects_to_latent(object_embeddings)
    
    def step(self, target_state: torch.Tensor, use_mppi: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Execute one planning step
        
        Returns: (action, info)
        """
        if use_mppi:
            action_sequence = self.planner.plan_mppi(
                self.current_latent, target_state
            )
        else:
            action_sequence = self.planner.plan(
                self.current_latent, target_state,
                num_iterations=80, lr=0.01
            )
        
        # Get first action (receding horizon)
        first_action = action_sequence[:, 0, :]
        
        # Simulate forward one step
        next_latent, uncertainty = self.world_model.predict(
            self.current_latent, first_action
        )
        
        # Update current state
        old_latent = self.current_latent
        self.current_latent = next_latent
        
        info = {
            "action_sequence": action_sequence,
            "uncertainty": uncertainty,
            "old_latent": old_latent,
            "new_latent": next_latent,
            "action_sequence_complete": action_sequence
        }
        
        return first_action, info
    
    def plan_full_trajectory(self, target_state: torch.Tensor, 
                             steps: int = 10) -> torch.Tensor:
        """
        Plan a full trajectory of actions
        """
        action_sequences = []
        current = self.current_latent
        
        for t in range(steps):
            # Plan from current state
            actions = self.planner.plan(current, target_state, num_iterations=50, lr=0.01)
            first_action = actions[:, 0, :]
            action_sequences.append(first_action)
            
            # Simulate forward
            current, _ = self.world_model.predict(current, first_action)
        
        return torch.stack(action_sequences, dim=1)


# Example usage: Full pipeline
if __name__ == "__main__":
    # Create world model
    world_model = WorldModel(latent_dim=64, action_dim=2, 
                            object_dim=32, max_objects=10)
    
    # Create planner
    planner = DifferentiablePlanner(world_model, horizon=8, action_dim=2)
    
    # Create controller
    controller = PlanningController(world_model, planner)
    
    # Generate sample object embeddings and targets
    batch_size = 1
    max_objects = 10
    object_dim = 32
    
    # Initial scene
    initial_objects = torch.randn(batch_size, max_objects, object_dim)
    controller.reset(initial_objects)
    
    # Target state (we want to reach a specific latent state)
    target = torch.randn(batch_size, 64)
    
    print("Starting planning...")
    
    # Plan and execute 10 steps
    for step in range(10):
        action, info = controller.step(target, use_mppi=False)
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action.squeeze().detach().numpy()}")
        print(f"  Uncertainty: {info['uncertainty'].item():.4f}")
        
        # Check if reached target
        dist = F.mse_loss(info['new_latent'], target).item()
        print(f"  Distance to target: {dist:.6f}")
        
        if dist < 0.01:
            print("  Target reached!")
            break
    
    print("\nComplete!")