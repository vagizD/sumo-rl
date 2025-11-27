import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import PyTorchObs, GymEnv
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import TrainFreq, TrainFrequencyUnit, polyak_update


class DummyPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor_class=None,
            features_extractor_kwargs={},
        )

        self.optimizers = []

    def _build(self, lr_schedule) -> None:
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError("DummyPolicy is not used for inference")

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, observation, deterministic=False):
        raise NotImplementedError


class QNet(nn.Module):
    def __init__(self, observation_space, action_space, hid_size):
        super().__init__()
        self.hid_size = hid_size
        self.q_net = nn.Sequential(
            nn.Linear(observation_space, self.hid_size), nn.ReLU(),
            nn.Linear(    self.hid_size, self.hid_size), nn.ReLU(),
            nn.Linear(    self.hid_size,  action_space)
        )

    def forward(self, obs):
        return self.q_net(obs)


class MyDQN(OffPolicyAlgorithm):
    def __init__(
            self,
            env: GymEnv,
            learning_rate: float = 1e-3,
            buffer_size: int = 100_000,
            batch_size: int = 128,
            hid_size: int = 128,
            gamma: float = 0.99,
            exploration_fraction: float = 0.3,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.02,
            target_update_interval: int = 2000,
            tau: float = 1.0,  # controls update rate of target network
            train_freq: int = 4,
            gradient_steps: int = 1,
            tensorboard_log: str = None,
            verbose: int = 0,
            device: str = "auto",
            seed: int | None = None,
    ):
        super().__init__(
            policy=DummyPolicy,
            env=env,
            policy_kwargs=None,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=TrainFreq(train_freq, TrainFrequencyUnit.STEP),
            gradient_steps=gradient_steps,
            tau=tau,  # controls soft/hard target update
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
        )

        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps

        self.hid_size = hid_size
        self.q_net = QNet(env.observation_space.shape[0], env.action_space.n, self.hid_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self._setup_model()

        self.target_update_interval = target_update_interval
        self.q_net_target = QNet(env.observation_space.shape[0], env.action_space.n, self.hid_size).to(self.device)

    def train(self, gradient_steps: int = 1, batch_size: int = 128) -> None:
        self._update_learning_rate(self.policy.optimizers)

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                next_q = self.q_net_target(replay_data.next_observations)
                next_q_max = next_q.max(dim=1)[0]
                target_q = replay_data.rewards.flatten() + (1 - replay_data.dones.flatten()) * self.gamma * next_q_max

            current_q = self.q_net(replay_data.observations)
            current_q = current_q.gather(1, replay_data.actions).squeeze(-1)

            loss = nn.functional.mse_loss(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.logger.record("train/loss", loss.item())

        # self.num_timesteps = # of env.step()
        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), tau=self.tau)
            self.logger.record("train/target_update", 1)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        action = self._predict(observation, deterministic=deterministic)
        return action, None

    def _predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        observation = torch.as_tensor(observation, device=self.device).float()
        with torch.no_grad():
            q_values = self.q_net(observation)
        if deterministic:
            action = q_values.argmax(dim=1).cpu().numpy()
        else:
            # eps-greedy
            eps = self.exploration_rate
            if np.random.rand() < eps:
                action = np.array([self.action_space.sample() for _ in range(observation.shape[0])])
            else:
                action = q_values.argmax(dim=1).cpu().numpy()
        return action

    @property
    def exploration_rate(self) -> float:
        progress = min(1.0, self.num_timesteps / (self.exploration_fraction * self._total_timesteps))
        return np.interp(progress, [0, 1], [self.exploration_initial_eps, self.exploration_final_eps])

    def learn(self, *args, **kwargs):
        return super().learn(*args, **kwargs)
