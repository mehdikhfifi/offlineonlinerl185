from tarfile import DIRTYPE
from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class IFQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor_flow,
        make_actor_flow_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        online_training: bool = False,
        num_samples: int = 32,
        expectile: float = 0.9,
        rho: float = 0.5,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.flow_actor = make_actor_flow(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.value = make_value(observation_shape)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.flow_actor_optimizer = make_actor_flow_optimizer(self.flow_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.value_optimizer = make_value_optimizer(self.value.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.num_samples = num_samples
        self.expectile = expectile

    @staticmethod
    def expectile_loss(adv: torch.Tensor, expectile: float) -> torch.Tensor:
        """
        Compute the expectile loss for IFQL
        """
        # DONE(student): Implement the expectile loss 
        abs = torch.where(adv > 0, (1.0 - expectile), expectile)
        return (abs * adv**2).mean()

    @torch.compile
    def update_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """
        Update value function
        """
        # TODO(student): Implement the value function update
        with torch.no_grad():
            q_vals = self.target_critic(observations, actions)
            q = q_vals.min(dim=0).values
        v = self.value(observations)

        loss = self.expectile_loss(v - q, self.expectile)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
        return {
            "v_loss": loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Rejection / best-of-n sampling using the flow policy and critic.

        We:
          1. Sample multiple candidate actions via the BC flow.
          2. Evaluate them with the critic.
          3. Pick the action with the highest Q-value.
        """
        batch_size = observations.shape[0]
        obs_dim = observations.shape[1]

        obs_tiled = observations.unsqueeze(dim=0).expand(self.num_samples, batch_size, obs_dim)
        obs_flat = obs_tiled.reshape(self.num_samples * batch_size, obs_dim)
        noise = torch.randn(
            (self.num_samples * batch_size, self.action_dim),
            device=observations.device,
            dtype=observations.dtype,
        )

        candidate_actions = self.get_flow_action(obs_flat, noise).reshape(
            self.num_samples, batch_size, self.action_dim
        )

        candidate_actions_flat = candidate_actions.reshape(self.num_samples * batch_size, self.action_dim)
        q_values = self.critic(obs_flat, candidate_actions_flat).reshape(-1, self.num_samples, batch_size)
        q_values = q_values.min(dim=0).values
        best_idx = q_values.argmax(dim=0)
        batch_idx = torch.arange(batch_size, device=observations.device)

        return candidate_actions[best_idx, batch_idx]

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        obs = ptu.from_numpy(observation[None])
        with torch.no_grad():
            action = self.sample_actions(obs)
        return ptu.to_numpy(action[0])

    @torch.compile
    def get_flow_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Compute the flow action using Euler integration for `self.flow_steps` steps.
        """
        # DONE(student): Implement euler integration to get flow action
        a = noise
        dt = 1.0 / self.flow_steps
        for k in range(self.flow_steps):
            t = torch.full(
                (*a.shape[:-1], 1),
                (k + 0.5) * dt,
                device=a.device,
                dtype=a.dtype,
            )
            v = self.flow_actor(observation, a, t)
            a = a + dt * v
        
        return torch.clamp(a, -1, 1)
        

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a) using the learned value function for bootstrapping,
        as in IFQL / IQL-style critic training.
        """
        # DONE(student): Implement Q-function update
        with torch.no_grad():
            v_next = self.value(next_observations)
            y = rewards.flatten() + (
                (1.0 - dones.flatten().float()) * self.discount * v_next
            )

        q = self.critic(observations, actions)
        loss = ((q - y) ** 2).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }


    @torch.compile
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the flow actor using the velocity matching loss.
        """
        # TODO(student): Implement flow actor update
        batch = observations.shape[0]
        z = torch.randn(batch, self.action_dim, device=observations.device, dtype=observations.dtype)
        t = torch.rand((batch, 1), device=observations.device, dtype=observations.dtype)
        a_tilde = (1 - t) * z + t * actions
        target = actions - z
        v_pred = self.flow_actor(observations, a_tilde, t)
        loss = ((v_pred - target) ** 2).mean()
        
        # TODO(student): Update flow actor
        self.flow_actor_optimizer.zero_grad()
        loss.backward()
        self.flow_actor_optimizer.step()
        
        return {
            "loss": loss
        }


    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_v = self.update_value(observations, actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"value/{k}": v.item() for k, v in metrics_v.items()},
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.target_update_rate * param.data + (1 - self.target_update_rate) * target_param.data
            )
