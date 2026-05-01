from tarfile import DIRTYPE
from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Sequence


class DSRLAgent(nn.Module):
    """DSRL agent - https://arxiv.org/abs/2506.15799"""

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_flow_actor,
        make_bc_flow_actor_optimizer,
        make_noise_actor,
        make_noise_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_z_critic,
        make_z_critic_optimizer,
        make_log_alpha,
        make_log_alpha_optimizer,


        discount: float,
        target_update_rate: float,
        flow_steps: int,
        noise_scale: float = 1.0,

        online_training: bool = False,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.noise_scale = noise_scale
        self.target_entropy = -action_dim

        # DONE(student): Create BC flow actor and target BC flow actor
        self.bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor.load_state_dict(self.bc_flow_actor.state_dict())
        # DONE(student): Create noise policy
        self.noise_actor = make_noise_actor(observation_shape, action_dim)
        # DONE(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions), and z critic (for noise policy)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.z_critic = make_z_critic(observation_shape, action_dim)
        # DONE(student): Create learnable entropy coefficient
        self.log_alpha = make_log_alpha()
        # DONE(student): Create optimizers for all the above models
        self.bc_flow_actor_optimizer = make_bc_flow_actor_optimizer(self.bc_flow_actor.parameters())
        self.noise_actor_optimizer = make_noise_actor_optimizer(self.noise_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.z_critic_optimizer = make_z_critic_optimizer(self.z_critic.parameters())
        self.alpha_optimizer = make_log_alpha_optimizer(self.log_alpha.parameters())
        
        self.to(ptu.device)

    @property
    def alpha(self):
        # TODO(student): Allow access to the learnable entropy coefficient (tip: if you are learning log alpha, as in HW3, then when we want to use alpha, you should return the exponential of the log alpha)
        return self.log_alpha()

    @torch.compiler.disable
    def sample_flow_actions(self, observations: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """Euler integration of BC flow from t=0 to t=1."""
        # DONE(student): Implement Euler integration of BC flow. Keep in mind that the target BC flow actor should be used
        # Also note that we can control what we use as the noise input (could be sampled from a noise policy or from a normal distribution)
        dt = 1 / self.flow_steps
        a = noises
        for k in range(self.flow_steps):
            t = torch.full(
                (*a.shape[:-1], 1),
                (k + 0.5) * dt,
                device=observations.device,
                dtype=observations.dtype
            )
            v = self.target_bc_flow_actor(observations, a, t)
            a = a + dt * v
        
        return torch.clamp(a, -1, 1)

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """Sample actions using noise policy for noise input to BC flow policy."""
        # DONE(student): Sample noise from the noise policy and use to sample actions from the BC flow policy
        noise_dist = self.noise_actor(observations)
        noises = self.noise_scale * noise_dist.sample()
        return self.sample_flow_actions(observations, noises)

    
    def get_action(self, observation: np.ndarray):
        """Used for evaluation."""
        # DONE(student): Implement get action
        obs = ptu.from_numpy(observation[None])
        action = self.sample_actions(obs)
        return ptu.to_numpy(action[0])

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """Update critic"""
        # DONE(student): Implement critic loss
        with torch.no_grad():
            next_actions = self.sample_actions(next_observations)
            next_q = self.target_critic(next_observations, next_actions).mean(dim=0)
            y = rewards.flatten() + (
                (1.0 - dones.flatten().float()) * self.discount * next_q
            )
        q = self.critic(observations, actions)

        loss = ((q - y) ** 2).mean()

        # DONE(student): Update critic
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }
    
    def update_qz(self, 
        observations: torch.Tensor,
    ) -> dict:
        """Update z_critic."""
        
        # DONE(student): Implement z_critic loss
        batch = observations.shape[0]
        with torch.no_grad():
            noises = torch.randn(batch, self.action_dim, device=observations.device, dtype=observations.dtype)
            bc_actions = self.sample_flow_actions(observations, noises)
            y = self.critic(observations, bc_actions).mean(dim=0)
        qz = self.z_critic(observations, noises)
        loss = ((y - qz) ** 2).mean()

        # DONE(student): Update z_critic
        self.z_critic_optimizer.zero_grad()
        loss.backward()
        self.z_critic_optimizer.step()
        
        return {
            "q_loss": loss,
            "q_mean": qz.mean(),
            "q_max": qz.max(),
            "q_min": qz.min(),
        }

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """Update BC flow actor"""
        # TODO(student): Implement BC flow loss
        batch = observations.shape[0]
        z = torch.randn(batch, self.action_dim, device=observations.device, dtype=observations.dtype)
        t = torch.rand((batch, 1), device=observations.device, dtype=observations.dtype)
        a_tilde = (1 - t) * z + t * actions
        target = actions - z
        v_pred = self.bc_flow_actor(observations, a_tilde, t)
        loss = ((v_pred - target) ** 2).mean()
        
        # TODO(student): Update BC flow actor
        self.bc_flow_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_flow_actor_optimizer.step()
        
        return {
            "loss": loss
        }
    
    def update_noise_actor(self,
        observations: torch.Tensor,
    ) -> dict:
        """Update noise actor."""
        # TODO(student): Implement noise actor loss
        noise_dist = self.noise_actor(observations)
        noises = noise_dist.rsample()
        log_probs = noise_dist.log_prob(noises)
        qz_loss = self.z_critic(observations, self.noise_scale * noises)
        loss = (self.alpha * log_probs - qz_loss).mean()
        
        # TODO(student): Update noise actor
        self.noise_actor_optimizer.zero_grad()
        loss.backward()
        self.noise_actor_optimizer.step()
        
        return {
            "loss": loss
        }

    def update_alpha(self, observations: torch.Tensor) -> dict:
        """Update alpha."""
        # DONE(student): Implement alpha loss
        noise_dist = self.noise_actor(observations)
        noises = noise_dist.rsample()
        log_probs = noise_dist.log_prob(noises)
        neg_entropy_loss = log_probs
        loss = -(self.alpha * (neg_entropy_loss + self.target_entropy)).mean()
        
        # DONE(student): Update alpha
        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()

        return {
            "loss": loss
        }

    def update_target_critic(self) -> None:
        # DONE(student): Implement target critic update
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(
                (1 - self.target_update_rate) * target_param.data + self.target_update_rate * param
            )

    def update_target_bc_flow_actor(self) -> None:
        # DONE(student): Implement target BC flow actor update
        for param, target_param in zip(self.bc_flow_actor.parameters(), self.target_bc_flow_actor.parameters()):
            target_param.data.copy_(
                (1 - self.target_update_rate) * target_param.data + self.target_update_rate * param
            )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        # DONE(student): Update critic, z_critic, actor, noise actor, and alpha - feel free to modify this code according to your setup!
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_qz = self.update_qz(observations)
        metrics_actor = self.update_actor(observations, actions)
        metrics_noise_actor = self.update_noise_actor(observations)
        metrics_alpha = self.update_alpha(observations)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"z_critic/{k}": v.item() for k, v in metrics_qz.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
            **{f"noise_actor/{k}": v.item() for k, v in metrics_noise_actor.items()},
            **{f"alpha/{k}": v.item() for k, v in metrics_alpha.items()},
        }

        self.update_target_critic()
        self.update_target_bc_flow_actor()

        return metrics

