from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        noise = torch.randn(1, self.action_dim, device=observation.device)
        action = self.onestep_actor(observation, noise)
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        action = noise
        dt = 1.0 / self.flow_steps
        for i in range(self.flow_steps):
            t = torch.full((*action.shape[:-1], 1), i * dt, device=action.device)
            velocity = self.bc_actor(observation, action, t)
            action = action + dt * velocity
        action = torch.clamp(action, -1, 1)
        return action

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        q = self.critic(observations, actions)  # (2, B)

        with torch.no_grad():
            noise = torch.randn(next_observations.shape[0], self.action_dim, device=next_observations.device)
            next_actions = torch.clamp(self.onestep_actor(next_observations, noise), -1, 1)
            target_q = self.target_critic(next_observations, next_actions).mean(dim=0)
            y = rewards + self.discount * (1 - dones) * target_q

        loss = ((q[0] - y) ** 2 + (q[1] - y) ** 2).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        z = torch.randn_like(actions)
        t = torch.rand(observations.shape[0], 1, device=observations.device)
        a_tilde = (1 - t) * z + t * actions
        target = actions - z
        predicted = self.bc_actor(observations, a_tilde, t)
        loss = ((predicted - target) ** 2).sum(dim=-1).mean() / self.action_dim

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        noise = torch.randn_like(actions)
        onestep_actions = self.onestep_actor(observations, noise)

        with torch.no_grad():
            bc_actions = self.get_bc_action(observations, noise)

        mse = ((onestep_actions - bc_actions) ** 2).sum(dim=-1).mean()
        distill_loss = self.alpha / self.action_dim * ((onestep_actions - bc_actions) ** 2).sum(dim=-1).mean()

        onestep_actions_clipped = torch.clamp(onestep_actions, -1, 1)
        q_loss = -self.critic(observations, onestep_actions_clipped).mean(dim=0).mean()

        loss = distill_loss + q_loss

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
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
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.target_update_rate * param.data + (1 - self.target_update_rate) * target_param.data
            )