"""
neural_symbolic.py - Neural-Symbolic System
Extracts objects, attributes, and relations from scene descriptions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class Object:
    """Symbolic object representation"""
    id: str
    category: str
    position: List[float]
    attributes: Dict[str, Any]
    embedding: torch.Tensor = None


@dataclass
class Relation:
    """Symbolic relation between objects"""
    subject: str
    predicate: str
    object: str
    confidence: float


class SceneParser:
    """Parses JSON scene graphs into symbolic objects and relations"""
    
    @staticmethod
    def parse(scene_graph: Dict[str, Any]) -> Tuple[List[Object], List[Relation]]:
        objects = []
        relations = []
        
        # Parse objects
        for obj_data in scene_graph.get("objects", []):
            obj = Object(
                id=obj_data.get("id", f"obj_{len(objects)}"),
                category=obj_data.get("name", "unknown"),
                position=obj_data.get("position", [0.5, 0.5]),
                attributes=obj_data.get("properties", {})
            )
            objects.append(obj)
        
        # Parse relations
        for rel_data in scene_graph.get("relations", []):
            rel = Relation(
                subject=rel_data.get("subject", ""),
                predicate=rel_data.get("predicate", "unknown"),
                object=rel_data.get("object", ""),
                confidence=rel_data.get("confidence", 0.8)
            )
            relations.append(rel)
        
        return objects, relations


class NeuralSymbolicEncoder(nn.Module):
    """
    Encodes symbolic objects and relations into differentiable embeddings
    """
    
    def __init__(self, object_dim: int = 64, relation_dim: int = 32, max_objects: int = 10):
        super().__init__()
        self.max_objects = max_objects
        self.object_dim = object_dim
        
        # Category embedding
        self.category_embedding = nn.Embedding(50, 16)  # 50 categories
        
        # Position encoder (x, y, radius)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Attribute encoder (key-value pairs)
        self.attr_encoder = nn.Sequential(
            nn.Linear(8, 16),  # 8 attribute slots
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Final object fusion
        self.object_fusion = nn.Sequential(
            nn.Linear(16 + 16 + 16, object_dim),
            nn.ReLU(),
            nn.Linear(object_dim, object_dim)
        )
        
        # Relation encoder
        self.relation_encoder = nn.Sequential(
            nn.Linear(2 * object_dim + 16, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, relation_dim)
        )
        
        # Relation predicate embedding
        self.predicate_embedding = nn.Embedding(30, 16)  # 30 relation types
        
    def encode_objects(self, objects: List[Object]) -> torch.Tensor:
        """
        Encode a list of objects into a tensor of shape [num_objects, object_dim]
        """
        if not objects:
            return torch.zeros(0, self.object_dim)
        
        batch_embeddings = []
        
        for obj in objects:
            # Category embedding
            cat_idx = hash(obj.category) % 50
            cat_emb = self.category_embedding(torch.tensor([cat_idx]))
            
            # Position embedding
            pos_vec = torch.tensor(obj.position + [0.08], dtype=torch.float32)  # radius
            pos_emb = self.position_encoder(pos_vec)
            
            # Attribute embedding
            attr_values = self._attributes_to_vector(obj.attributes)
            attr_emb = self.attr_encoder(attr_values)
            
            # Fuse into object embedding
            fused = torch.cat([cat_emb.squeeze(0), pos_emb, attr_emb])
            obj_emb = self.object_fusion(fused)
            batch_embeddings.append(obj_emb)
        
        return torch.stack(batch_embeddings)
    
    def encode_relations(self, objects: List[Object], relations: List[Relation]) -> torch.Tensor:
        """
        Encode relations into a tensor [num_relations, relation_dim]
        """
        if not relations:
            return torch.zeros(0, 32)
        
        # Build object ID -> index mapping
        obj_map = {obj.id: idx for idx, obj in enumerate(objects)}
        obj_embs = self.encode_objects(objects)
        
        rel_embeddings = []
        
        for rel in relations:
            sub_idx = obj_map.get(rel.subject, 0)
            obj_idx = obj_map.get(rel.object, 0)
            
            # Subject and object embeddings
            sub_emb = obj_embs[sub_idx]
            obj_emb = obj_embs[obj_idx]
            
            # Predicate embedding
            pred_idx = hash(rel.predicate) % 30
            pred_emb = self.predicate_embedding(torch.tensor([pred_idx])).squeeze(0)
            
            # Combine
            combined = torch.cat([sub_emb, obj_emb, pred_emb])
            rel_emb = self.relation_encoder(combined)
            rel_embeddings.append(rel_emb)
        
        return torch.stack(rel_embeddings)
    
    def _attributes_to_vector(self, attributes: Dict[str, Any]) -> torch.Tensor:
        """Convert attribute dict to fixed-length vector"""
        vec = torch.zeros(8)
        for i, (key, value) in enumerate(attributes.items()):
            if i >= 8:
                break
            if isinstance(value, (int, float)):
                vec[i] = float(value)
            elif isinstance(value, str):
                vec[i] = float(hash(value) % 100) / 100.0
        return vec


class SymbolicReasoner(nn.Module):
    """
    Neural-symbolic reasoner that performs differentiable logic inference
    """
    
    def __init__(self, object_dim: int = 64, relation_dim: int = 32):
        super().__init__()
        self.object_dim = object_dim
        self.relation_dim = relation_dim
        
        # Rule weights for common predicates
        self.rule_weights = nn.Parameter(torch.randn(20, 2 * object_dim + 16))
        
        # Inference network
        self.inference_net = nn.Sequential(
            nn.Linear(object_dim * 3 + relation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def infer_relation(self, subj_emb: torch.Tensor, obj_emb: torch.Tensor, 
                       relation_type: str) -> torch.Tensor:
        """
        Infer probability of a relation between two objects
        Returns: confidence [0, 1]
        """
        # Predicate embedding
        pred_idx = hash(relation_type) % 30
        pred_emb = torch.zeros(16)
        pred_emb[pred_idx % 16] = 1.0
        
        combined = torch.cat([subj_emb, obj_emb, pred_emb])
        confidence = torch.sigmoid(self.inference_net(combined))
        return confidence.squeeze()
    
    def forward(self, objects: List[Object], relations: List[Relation]) -> Dict[str, Any]:
        """
        Perform symbolic reasoning and return inferred facts
        """
        encoder = NeuralSymbolicEncoder()
        obj_embs = encoder.encode_objects(objects)
        
        inferred_facts = []
        
        # For each pair of objects, infer possible relations
        for i, obj_i in enumerate(objects):
            for j, obj_j in enumerate(objects):
                if i == j:
                    continue
                
                # Infer containment
                containment = self.infer_relation(obj_embs[i], obj_embs[j], "contains")
                if containment > 0.5:
                    inferred_facts.append({
                        "subject": obj_i.id,
                        "predicate": "contains",
                        "object": obj_j.id,
                        "confidence": float(containment)
                    })
                
                # Infer support (on top of)
                support = self.infer_relation(obj_embs[i], obj_embs[j], "supports")
                if support > 0.5:
                    inferred_facts.append({
                        "subject": obj_i.id,
                        "predicate": "supports",
                        "object": obj_j.id,
                        "confidence": float(support)
                    })
        
        return {
            "object_embeddings": obj_embs,
            "inferred_facts": inferred_facts,
            "num_objects": len(objects)
        }


# Example usage
if __name__ == "__main__":
    # Test scene
    scene = {
        "objects": [
            {"id": "box1", "name": "box", "position": [0.3, 0.4], "properties": {"color": "red"}},
            {"id": "ball1", "name": "ball", "position": [0.5, 0.6], "properties": {"color": "blue"}},
            {"id": "target1", "name": "target", "position": [0.8, 0.7], "properties": {}}
        ],
        "relations": [
            {"subject": "ball1", "predicate": "near", "object": "box1", "confidence": 0.9}
        ]
    }
    
    objects, relations = SceneParser.parse(scene)
    encoder = NeuralSymbolicEncoder()
    obj_embs = encoder.encode_objects(objects)
    rel_embs = encoder.encode_relations(objects, relations)
    reasoner = SymbolicReasoner()
    result = reasoner.forward(objects, relations)
    
    print(f"Parsed {len(objects)} objects and {len(relations)} relations")
    print(f"Object embeddings shape: {obj_embs.shape}")
    print(f"Relation embeddings shape: {rel_embs.shape}")
    print(f"Inferred facts: {result['inferred_facts']}")