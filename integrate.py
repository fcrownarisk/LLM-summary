"""
integrate.py - Complete pipeline integration
Connects neural-symbolic, world model, and differentiable planner
"""

import torch
from neural_symbolic import SceneParser, NeuralSymbolicEncoder, SymbolicReasoner
from world_model import WorldModel
from differentiable_planner import DifferentiablePlanner, PlanningController

def full_pipeline(scene_json: dict, target_description: str):
    """
    Complete pipeline:
    1. Parse scene → objects + relations
    2. Neural-symbolic encoding
    3. World model simulation
    4. Differentiable planning
    """
    
    # 1. Parse scene
    objects, relations = SceneParser.parse(scene_json)
    print(f"Parsed {len(objects)} objects")
    print(f"Parsed {len(relations)} relations")
    
    # 2. Encode objects to embeddings
    encoder = NeuralSymbolicEncoder(object_dim=32)
    obj_embs = encoder.encode_objects(objects)
    rel_embs = encoder.encode_relations(objects, relations)
    print(f"Object embeddings: {obj_embs.shape}")
    
    # 3. Reason about the scene
    reasoner = SymbolicReasoner()
    reasoning_result = reasoner.forward(objects, relations)
    print(f"Inferred facts: {len(reasoning_result['inferred_facts'])}")
    
    # 4. Prepare for world model
    max_objects = 10
    object_dim = 32
    batch_size = 1
    
    # Pad object embeddings to max_objects
    padded_embs = torch.zeros(batch_size, max_objects, object_dim)
    padded_embs[0, :obj_embs.shape[0], :] = obj_embs
    
    # 5. World model
    world_model = WorldModel(latent_dim=64, action_dim=2,
                             object_dim=object_dim, max_objects=max_objects)
    
    # Encode to latent
    latent = world_model.encode_objects_to_latent(padded_embs)
    print(f"Latent state: {latent.shape}")
    
    # 6. Define target (use reasoning result)
    target_latent = torch.randn(batch_size, 64)  # In practice, derive from target_description
    
    # 7. Differentiable planning
    planner = DifferentiablePlanner(world_model, horizon=10, action_dim=2)
    controller = PlanningController(world_model, planner)
    controller.current_latent = latent
    
    print("\nPlanning optimal trajectory...")
    actions, info = controller.step(target_latent, use_mppi=False)
    
    print(f"Optimal action: {actions.squeeze().detach().numpy()}")
    print(f"Uncertainty: {info['uncertainty'].item():.4f}")
    
    return {
        "objects": objects,
        "relations": relations,
        "obj_embeddings": obj_embs,
        "reasoning": reasoning_result,
        "latent_state": latent,
        "action": actions,
        "info": info
    }


if __name__ == "__main__":
    # Example scene
    scene = {
        "objects": [
            {"id": "robot", "name": "robot", "position": [0.2, 0.3], "properties": {"color": "silver"}},
            {"id": "block", "name": "block", "position": [0.5, 0.5], "properties": {"color": "red"}},
            {"id": "target", "name": "target", "position": [0.8, 0.7], "properties": {}}
        ],
        "relations": [
            {"subject": "block", "predicate": "near", "object": "robot", "confidence": 0.8}
        ]
    }
    
    result = full_pipeline(scene, "Move robot to target position")
    
    print("\n" + "="*50)
    print("Pipeline complete!")
    print("="*50)