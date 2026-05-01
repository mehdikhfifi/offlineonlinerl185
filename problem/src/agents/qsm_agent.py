from re import T
from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu
import math
from typing import Callable, Optional, Sequence, Tuple, List

class QSMAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        inv_temp: float,
        flow_steps: int,
    ):
        super().__init__()

        self.action_dim = action_dim
        
        # TODO(student): Create actor
        self.actor = make_actor(observation_shape, action_dim)
        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        # TODO(student): Create optimizers for all the above models
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.inv_temp = inv_temp
        self.flow_steps = flow_steps

        betas = self.cosine_beta_schedule(flow_steps)
        alphas = 1.0 - betas
        alpha_hats = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas) # DONE(student): Implement betas
        self.register_buffer("alphas", alphas) # DONE(student): Implement alphas
        self.register_buffer("alpha_hats", alpha_hats) # DONE(student): Implement alpha_hats

        self.to(ptu.device)
    
    def cosine_beta_schedule(self, timesteps):
        """
        Cosine annealing beta schedule
        """
        # DONE(student): Implement cosine annealing beta schedule
        s = 0.08
        t = torch.arange(timesteps + 1, dtype=torch.float32)
        alpha_bar = torch.cos(((t / timesteps + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(1e-5, 0.999)
    
    @torch.compiler.disable
    def ddpm_sampler(self, observations: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        DDPM sampling
        """
        # DONE(student): Implement DDPM sampling
        x_t = noise
        betas = self.betas.view(-1)
        alphas = self.alphas.view(-1)
        alpha_hats = self.alpha_hats.view(-1)

        for t in reversed(range(self.flow_steps)):
            if t > 0:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)

            alpha_t = alphas[t]
            alpha_hat_t = alpha_hats[t]
            beta_t = betas[t]
            alpha_hat_prev = alpha_hats[t - 1] if t > 0 else torch.tensor(
                1.0, device=x_t.device, dtype=x_t.dtype
            )

            beta_tilde = ((1.0 - alpha_hat_prev) / (1.0 - alpha_hat_t)) * beta_t

            t_tensor = torch.full(
                (*x_t.shape[:-1], 1),
                float(t),
                device=x_t.device,
                dtype=x_t.dtype,
            )
            epsilon_pred = self.actor(observations, x_t, t_tensor)

            mu = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_hat_t)) * epsilon_pred
            )
            x_t = mu + torch.sqrt(beta_tilde) * z

        return torch.clamp(x_t, -1.0, 1.0)
    
    def get_action(self, observation: torch.Tensor):
        """
        Used for evaluation.
        """
        # DONE(student): Implement get_action
        with torch.no_grad():
            obs = ptu.from_numpy(observation[None]) if isinstance(observation, np.ndarray) else observation[None]
            init_noise = torch.randn((1, self.action_dim), device=obs.device, dtype=obs.dtype)
            action = self.ddpm_sampler(obs, init_noise)
            return ptu.to_numpy(action[0])




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
        Update Critic
        """
        # DONE(student): Implement critic update
        with torch.no_grad():
            noise = torch.randn(
                (next_observations.shape[0], self.action_dim),
                device=next_observations.device,
                dtype=next_observations.dtype,
            )
            next_actions = self.ddpm_sampler(next_observations, noise)
            target_q = self.target_critic(next_observations, next_actions).min(dim=0).values
            y = rewards.flatten() + self.discount * (1.0 - dones.flatten().float()) * target_q

        q = self.critic(observations, actions)
        loss = ((q - y.unsqueeze(0)) ** 2).mean()

        # DONE(student): Update critic
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }
        
    @torch.compiler.disable
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the actor
        """
        # DONE(student): Implement actor update
        batch_size = observations.shape[0]
        z = torch.randn(
            (batch_size, self.action_dim),
            device=observations.device,
            dtype=observations.dtype,
        )
        t_idx = torch.randint(
            low=0,
            high=self.flow_steps,
            size=(batch_size,),
            device=observations.device,
        )

        alpha_hat_t = self.alpha_hats.view(-1)[t_idx].unsqueeze(-1)
        a_tilde = torch.sqrt(alpha_hat_t) * actions + torch.sqrt(1.0 - alpha_hat_t) * z

        t = t_idx.to(dtype=observations.dtype).unsqueeze(-1)
        eps_pred = self.actor(observations, a_tilde, t)

        actions_for_grad = actions.detach().requires_grad_(True)
        q_for_grad = self.critic(observations, actions_for_grad).mean(dim=0)
        q_grad = torch.autograd.grad(q_for_grad.sum(), actions_for_grad, create_graph=False)[0].detach()

        qsm_loss = ((eps_pred - self.inv_temp * q_grad) ** 2).mean()
        ddpm_loss = ((z - eps_pred) ** 2).mean()
        loss = qsm_loss + self.alpha * ddpm_loss

        # DONE(student): Update actor
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "loss": loss,
            "qsm_loss": qsm_loss,
            "ddpm_loss": ddpm_loss,
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
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # DONE(student): Update target_critic using Polyak averaging with self.target_update_rate
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(
                (1.0 - self.target_update_rate) * target_param.data
                + self.target_update_rate * param.data
            )