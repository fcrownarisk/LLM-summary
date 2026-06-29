#!/usr/bin/env python3
"""
LLM Physical World Manipulator
================================
A self-contained system that uses DeepSeek-VL for perception,
a 2D physics engine for world modeling, and MPC for control.

Usage:
    python llm_physical_agent.py [--mock] [--steps 20]

Dependencies:
    pip install numpy torch transformers pillow
"""

import json
import re
import argparse
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image


@dataclass
class Particle:
    pos: np.ndarray
    vel: np.ndarray
    radius: float = 0.08
    mass: float = 1.0


class PhysicsEngine:
    def __init__(self, gravity: Tuple[float, float] = (0.0, 0.0),
                 damping: float = 0.98, dt: float = 0.05):
        self.gravity = np.array(gravity, dtype=np.float32)
        self.damping = damping
        self.dt = dt
        self.particles: List[Particle] = []
        self.bounds = [0.0, 1.0, 0.0, 1.0]

    def set_particles(self, positions: List[List[float]], radii: Optional[List[float]] = None):
        self.particles = []
        for i, pos in enumerate(positions):
            r = radii[i] if radii and i < len(radii) else 0.08
            self.particles.append(Particle(
                pos=np.array(pos, dtype=np.float32),
                vel=np.array([0.0, 0.0], dtype=np.float32),
                radius=r
            ))

    def step(self, actions: Optional[List[List[float]]] = None):
        if actions is None:
            actions = [[0.0, 0.0] for _ in self.particles]

        for i, p in enumerate(self.particles):
            force = self.gravity * p.mass + np.array(actions[i], dtype=np.float32)
            acc = force / p.mass

            p.vel += acc * self.dt
            p.vel *= self.damping
            p.pos += p.vel * self.dt

            if p.pos[0] - p.radius < self.bounds[0]:
                p.pos[0] = self.bounds[0] + p.radius
                p.vel[0] = -p.vel[0] * 0.7
            if p.pos[0] + p.radius > self.bounds[1]:
                p.pos[0] = self.bounds[1] - p.radius
                p.vel[0] = -p.vel[0] * 0.7
            if p.pos[1] - p.radius < self.bounds[2]:
                p.pos[1] = self.bounds[2] + p.radius
                p.vel[1] = -p.vel[1] * 0.7
            if p.pos[1] + p.radius > self.bounds[3]:
                p.pos[1] = self.bounds[3] - p.radius
                p.vel[1] = -p.vel[1] * 0.7

        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                self._resolve_collision(self.particles[i], self.particles[j])

    def _resolve_collision(self, p1: Particle, p2: Particle):
        delta = p1.pos - p2.pos
        dist = np.linalg.norm(delta)
        min_dist = p1.radius + p2.radius
        if dist < min_dist and dist > 0.001:
            overlap = (min_dist - dist) / 2
            direction = delta / dist
            p1.pos += direction * overlap
            p2.pos -= direction * overlap
            rel_vel = (p1.vel - p2.vel).dot(direction)
            if rel_vel < 0:
                impulse = rel_vel / (1.0 / p1.mass + 1.0 / p2.mass) * 0.5
                p1.vel -= impulse / p1.mass * direction
                p2.vel += impulse / p2.mass * direction

    def get_state_vector(self) -> np.ndarray:
        state = []
        for p in self.particles:
            state.extend([p.pos[0], p.pos[1], p.vel[0], p.vel[1]])
        return np.array(state, dtype=np.float32)

    def set_state_vector(self, state: np.ndarray):
        idx = 0
        for p in self.particles:
            p.pos[0] = state[idx]
            p.pos[1] = state[idx + 1]
            p.vel[0] = state[idx + 2]
            p.vel[1] = state[idx + 3]
            idx += 4

    def clone_and_simulate(self, state: np.ndarray, actions: List[List[float]],
                           horizon: int) -> np.ndarray:
        saved_state = self.get_state_vector().copy()
        self.set_state_vector(state.copy())

        for t in range(horizon):
            act = actions[t] if t < len(actions) else [0.0, 0.0]
            action_list = [[0.0, 0.0] for _ in self.particles]
            if action_list:
                action_list[0] = act
            self.step(action_list)

        final_state = self.get_state_vector().copy()
        self.set_state_vector(saved_state)
        return final_state

    def get_positions(self) -> List[List[float]]:
        return [[p.pos[0], p.pos[1]] for p in self.particles]


class DeepSeekVLPerceptor:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-vl-7b-base",
                 device: str = "cuda", use_mock: bool = False):
        self.use_mock = use_mock
        if not use_mock:
            try:
                from transformers import AutoProcessor, AutoModelForCausalLM
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=device
                )
            except Exception as e:
                self.use_mock = True

    def perceive(self, image_path: str, instruction: str = "") -> Dict[str, Any]:
        if self.use_mock:
            return self._mock_perceive()

        try:
            image = Image.open(image_path).convert("RGB")

            if not instruction:
                instruction = (
                    "Analyze this 2D scene. Output ONLY a JSON object with this exact format: "
                    '{"objects": [{"id": "obj1", "name": "block", "position": [0.5, 0.3], "radius": 0.08}], '
                    '"target": {"position": [0.8, 0.6]}, "obstacles": [{"id": "obs1", "position": [0.4, 0.4]}]}'
                    "Use normalized coordinates in [0, 1] range."
                )

            prompt = f"<image>\n{instruction}"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False
                )

            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            return self._parse_json(response)

        except Exception as e:
            return self._mock_perceive()

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass
        return self._mock_perceive()

    def _mock_perceive(self) -> Dict[str, Any]:
        x, y = np.random.uniform(0.2, 0.8, 2).tolist()
        tx, ty = np.random.uniform(0.2, 0.8, 2).tolist()
        return {
            "objects": [{"id": "robot", "position": [x, y], "radius": 0.08}],
            "target": {"position": [tx, ty]},
            "obstacles": []
        }


class SceneBridge:
    @staticmethod
    def scene_to_state(scene: Dict[str, Any], num_particles: int = 1) -> np.ndarray:
        state = []
        objects = scene.get("objects", [{"position": [0.5, 0.5]}])

        for i in range(num_particles):
            if i < len(objects):
                pos = objects[i].get("position", [0.5, 0.5])
            else:
                pos = [0.5, 0.5]
            state.extend([pos[0], pos[1], 0.0, 0.0])

        return np.array(state, dtype=np.float32)

    @staticmethod
    def get_target(scene: Dict[str, Any]) -> np.ndarray:
        target = scene.get("target", {"position": [0.9, 0.9]})
        pos = target.get("position", [0.9, 0.9])
        return np.array([pos[0], pos[1]], dtype=np.float32)


class MPCPlanner:
    def __init__(self, physics: PhysicsEngine, horizon: int = 10,
                 num_samples: int = 50, action_scale: float = 1.5):
        self.physics = physics
        self.horizon = horizon
        self.num_samples = num_samples
        self.action_scale = action_scale

    def plan(self, current_state: np.ndarray, target_position: np.ndarray) -> np.ndarray:
        best_action = np.array([0.0, 0.0])
        best_cost = float('inf')
        target_pos = target_position.copy()

        for _ in range(self.num_samples):
            actions = np.random.uniform(-self.action_scale, self.action_scale,
                                       (self.horizon, 2)).tolist()

            final_state = self.physics.clone_and_simulate(
                state=current_state,
                actions=actions,
                horizon=self.horizon
            )

            final_pos = final_state[0:2]
            pos_cost = np.linalg.norm(final_pos - target_pos) * 2.0
            effort_cost = np.mean(np.abs(actions)) * 0.1
            cost = pos_cost + effort_cost

            if cost < best_cost:
                best_cost = cost
                best_action = np.array(actions[0])

        return best_action


class LLMPhysicalAgent:
    def __init__(self, use_mock: bool = False, horizon: int = 10,
                 num_samples: int = 50):
        self.perceptor = DeepSeekVLPerceptor(use_mock=use_mock)
        self.physics = PhysicsEngine(gravity=(0.0, 0.0))
        self.bridge = SceneBridge()
        self.planner = MPCPlanner(
            physics=self.physics,
            horizon=horizon,
            num_samples=num_samples
        )
        self.step_count = 0

    def reset(self):
        self.physics.set_particles([[0.5, 0.5]])
        self.step_count = 0

    def step(self, image_path: str) -> Dict[str, Any]:
        self.step_count += 1

        scene = self.perceptor.perceive(image_path)

        current_state = self.bridge.scene_to_state(scene, num_particles=1)
        target_pos = self.bridge.get_target(scene)

        self.physics.set_particles([[current_state[0], current_state[1]]])
        self.physics.particles[0].vel = np.array([current_state[2], current_state[3]])
        physics_state = self.physics.get_state_vector()

        action = self.planner.plan(physics_state, target_pos)

        self.physics.step(actions=[[action[0], action[1]]])

        current_pos = self.physics.get_positions()[0]
        distance = np.linalg.norm(np.array(current_pos) - target_pos)
        reached = distance < 0.05

        return {
            "step": self.step_count,
            "scene": scene,
            "position": current_pos,
            "target": target_pos.tolist(),
            "action": action.tolist(),
            "distance": distance,
            "reached": reached
        }

    def run(self, image_path: str, max_steps: int = 30) -> bool:
        self.reset()
        for step in range(max_steps):
            status = self.step(image_path)
            if status["reached"]:
                return True
        return False


def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered physical world manipulation agent"
    )
    parser.add_argument("--mock", action="store_true",
                        help="Use mock perception (no GPU required)")
    parser.add_argument("--image", type=str, default="scene.png",
                        help="Path to input image (ignored in mock mode)")
    parser.add_argument("--steps", type=int, default=30,
                        help="Maximum number of control steps")
    parser.add_argument("--horizon", type=int, default=10,
                        help="MPC planning horizon (in frames)")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of MPC random samples")
    args = parser.parse_args()

    agent = LLMPhysicalAgent(
        use_mock=args.mock,
        horizon=args.horizon,
        num_samples=args.samples
    )

    success = agent.run(args.image, max_steps=args.steps)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())