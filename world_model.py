"""
world_model.py - World Model
Simulates physical dynamics in latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class LatentState:
    """Represents a state in latent space"""
    def __init__(self, state_vector: torch.Tensor, object_ids: List[str] = None):
        self.vector = state_vector
        self.object_ids = object_ids or []
        self.dim = state_vector.shape[-1]
    
    def clone(self):
        return LatentState(self.vector.clone(), self.object_ids.copy())


class TransitionModel(nn.Module):
    """
    Learned transition model: p(z_{t+1} | z_t, a_t)
    Predicts next latent state given current state and action
    """
    
    def __init__(self, latent_dim: int = 64, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Predict next state
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Uncertainty prediction (for planning under uncertainty)
        self.uncertainty = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Positive uncertainty
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state and uncertainty
        state: [batch, latent_dim]
        action: [batch, action_dim]
        Returns: (next_state, uncertainty)
        """
        combined = torch.cat([state, action], dim=-1)
        next_state = self.transition(combined)
        uncertainty = self.uncertainty(combined)
        return next_state, uncertainty
    
    def sample_rollout(self, initial_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Rollout a sequence of actions
        initial_state: [batch, latent_dim]
        actions: [batch, horizon, action_dim]
        Returns: [batch, horizon, latent_dim]
        """
        states = [initial_state]
        current = initial_state
        
        for t in range(actions.shape[1]):
            action = actions[:, t, :]
            next_state, _ = self.forward(current, action)
            states.append(next_state)
            current = next_state
        
        return torch.stack(states[1:], dim=1)


class ObservationDecoder(nn.Module):
    """
    Decodes latent state back to interpretable representation
    """
    
    def __init__(self, latent_dim: int = 64, output_dim: int = 16):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state to observable features
        Returns: [batch, output_dim] where output_dim encodes position + attributes
        """
        return self.decoder(latent_state)


class WorldModel(nn.Module):
    """
    Complete world model: latent dynamics + decoding
    """
    
    def __init__(self, latent_dim: int = 64, action_dim: int = 2, 
                 object_dim: int = 32, max_objects: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.max_objects = max_objects
        
        # State encoder: scene → latent
        self.state_encoder = nn.Sequential(
            nn.Linear(object_dim * max_objects + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Transition model
        self.transition = TransitionModel(latent_dim, action_dim)
        
        # Decoder: latent → object states
        self.object_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, object_dim * max_objects)
        )
        
        # Relation decoder
        self.relation_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def encode_objects_to_latent(self, object_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode object embeddings to latent state
        object_embeddings: [batch, max_objects, object_dim]
        """
        batch_size = object_embeddings.shape[0]
        flat = object_embeddings.view(batch_size, -1)
        
        # Append a small context vector
        context = torch.zeros(batch_size, 16, device=flat.device)
        combined = torch.cat([flat, context], dim=-1)
        
        return self.state_encoder(combined)
    
    def predict(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next latent state and uncertainty
        """
        return self.transition(latent_state, action)
    
    def decode_to_objects(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state to object embeddings
        Returns: [batch, max_objects, object_dim]
        """
        flat = self.object_decoder(latent_state)
        return flat.view(flat.shape[0], self.max_objects, -1)
    
    def simulate_rollout(self, initial_latent: torch.Tensor, 
                         action_sequence: torch.Tensor) -> torch.Tensor:
        """
        Simulate full trajectory in latent space
        initial_latent: [batch, latent_dim]
        action_sequence: [batch, horizon, action_dim]
        Returns: [batch, horizon, latent_dim]
        """
        return self.transition.sample_rollout(initial_latent, action_sequence)
    
    def forward(self, object_embeddings: torch.Tensor, 
                action_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode, simulate, decode
        """
        # Encode to latent
        latent = self.encode_objects_to_latent(object_embeddings)
        
        # Simulate
        trajectory = self.simulate_rollout(latent, action_sequence)
        
        # Decode each step
        batch_size = trajectory.shape[0]
        horizon = trajectory.shape[1]
        
        decoded_objects = []
        decoded_relations = []
        
        for t in range(horizon):
            latent_t = trajectory[:, t, :]
            obj_t = self.decode_to_objects(latent_t)
            decoded_objects.append(obj_t)
        
        return {
            "latent_trajectory": trajectory,
            "decoded_objects": torch.stack(decoded_objects, dim=1),
            "initial_latent": latent
        }


# Example usage
if __name__ == "__main__":
    batch_size = 4
    max_objects = 10
    object_dim = 64
    horizon = 8
    action_dim = 2
    latent_dim = 64
    
    # Create world model
    world_model = WorldModel(latent_dim=latent_dim, action_dim=action_dim,
                             object_dim=object_dim, max_objects=max_objects)
    
    # Sample inputs
    object_embs = torch.randn(batch_size, max_objects, object_dim)
    actions = torch.randn(batch_size, horizon, action_dim)
    
    # Forward pass
    result = world_model(object_embs, actions)
    
    print(f"Latent trajectory shape: {result['latent_trajectory'].shape}")
    print(f"Decoded objects shape: {result['decoded_objects'].shape}")
    print(f"Initial latent shape: {result['initial_latent'].shape}")
    
    # Test single step prediction
    latent = result['initial_latent']
    action = actions[:, 0, :]
    next_latent, uncertainty = world_model.predict(latent, action)
    print(f"Next latent shape: {next_latent.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")