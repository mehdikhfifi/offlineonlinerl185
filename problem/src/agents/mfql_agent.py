from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List

# multi step flow q learning


class MFQLAgent(nn.Module):
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
        k: int,
    ):
        super().__init__()

        assert k >= 2, "k must be at least 2"
        assert k <= flow_steps, f"k ({k}) must be <= flow_steps ({flow_steps})"

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)


        #
        self.actors = nn.ModuleList([
            make_onestep_actor(observation_shape, action_dim) for _ in range(k)
        ])
        #


        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.global_actor_optimizer = make_onestep_actor_optimizer(
            [p for p in self.actors.parameters()]
        )
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())


        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha
        self.k = k

        # split flow_steps into k segments; last actor's segment absorbs the remainder
        seg = flow_steps // k
        self._segment_ends = [(j + 1) * seg for j in range(k - 1)] + [flow_steps]




    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        noise = torch.randn(1, self.action_dim, device=observation.device)
        action = self.forward_actors(observation, noise)
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]




    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor, start: int):
        """
        Used for training.
        """
        action = noise
        dt = 1.0 / self.flow_steps
        # find which segment `start` falls in, integrate to that segment's end
        end = self.flow_steps
        for boundary in self._segment_ends:
            if start < boundary:
                end = boundary
                break
        for i in range(start, end):
            t = torch.full((*action.shape[:-1], 1), i * dt, device=action.device)
            velocity = self.bc_actor(observation, action, t)
            action = action + dt * velocity
        action = torch.clamp(action, -1, 1)
        return action




    def forward_actors(self, observations, noise):
        actions = noise
        for i in range(self.k):
            actions = self.actors[i](observations, actions)
            if i == len(self.actors) - 1:
                actions = torch.clamp(actions, -1, 1)
        return actions




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
            next_actions = self.forward_actors(next_observations, noise)
            target_q = self.target_critic(next_observations, next_actions).mean(dim=0)
            y = rewards + self.discount * (1 - dones.int()) * target_q

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




    def update_onestep_actor(self, observations, actions):
        noise = torch.randn_like(actions)

        actor_outputs = []
        prev = noise
        for i in range(self.k):
            inp = prev if i == 0 else prev.detach()
            out = self.actors[i](observations, inp)
            actor_outputs.append(out)
            prev = out

        with torch.no_grad():
            a = noise
            dt = 1.0 / self.flow_steps
            boundaries = set(self._segment_ends)
            targets = []
            for i in range(self.flow_steps):
                t = torch.full((*a.shape[:-1], 1), i * dt, device=a.device)
                a = a + dt * self.bc_actor(observations, a, t)
                if (i + 1) in boundaries:
                    targets.append(torch.clamp(a, -1, 1))

        mses = [((actor_outputs[i] - targets[i]) ** 2).sum(-1).mean() for i in range(self.k)]
        mse = sum(mses)

        distill_loss = (self.alpha / self.action_dim / self.k) * mse
        final_actions_clipped = torch.clamp(actor_outputs[-1], -1, 1)

        q_loss = -self.critic(observations, final_actions_clipped).mean(0).mean()

        loss = distill_loss + q_loss
        self.global_actor_optimizer.zero_grad()
        loss.backward()
        self.global_actor_optimizer.step()

        metrics = {
            "total_loss": loss, "distill_loss": distill_loss, "q_loss": q_loss,
            "mse": mse,
        }
        for i, m in enumerate(mses):
            metrics[f"mse_{i}"] = m
        return metrics




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

        def compute_norm(model):
            total_norm = 0
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            return total_norm

        with torch.no_grad():
            actor_norms = [compute_norm(actor) for actor in self.actors]

        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
            **{f"grad_norm/actor_{i}_grad_norms": n for i, n in enumerate(actor_norms)},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.target_update_rate * param.data + (1 - self.target_update_rate) * target_param.data
            )
