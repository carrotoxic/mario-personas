"""Clipped-surrogate PPO update with per-epoch target-KL early stopping."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn, optim

from src.models import actor_critic
from src.training import config
from src.training import rollouts

ExtraLossFn = Callable[[rollouts.ObsTensorDict, torch.Tensor], torch.Tensor]
"""Maps ``(minibatch_obs, minibatch_actions)`` to an additional scalar loss
term (already scaled by its coefficient), e.g. the DRAIL post-training
reference-policy KL penalty."""


def ppo_update(
    agent: actor_critic.ActorCritic,
    optimizer: optim.Optimizer,
    batch: rollouts.RolloutBatch,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: config.TrainConfig,
    *,
    extra_loss_fn: ExtraLossFn | None = None,
) -> dict[str, float]:
    """Run one full PPO update over a rollout and return loss diagnostics.
    Minibatch indices are shuffled with the global NumPy RNG (seeded once at startup);
    the epoch loop breaks early when the last minibatch's KL exceeds ``target_kl``."""
    b_obs = {key: value.reshape((-1,) + value.shape[2:]) for key, value in batch.obs.items()}
    b_logprobs = batch.logprobs.reshape(-1)
    b_actions = batch.actions.reshape((-1,) + batch.actions.shape[2:])
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = batch.values.reshape(-1)

    b_inds = np.arange(cfg.batch_size)
    clipfracs: list[float] = []
    extra_loss_value: float | None = None
    for _ in range(cfg.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, cfg.batch_size, cfg.minibatch_size):
            end = start + cfg.minibatch_size
            mb_inds = b_inds[start:end]
            mb_obs = rollouts.index_obs(b_obs, mb_inds)
            mb_actions = b_actions.long()[mb_inds]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

            mb_advantages = b_advantages[mb_inds]
            if cfg.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(-1)
            # clip_coef doubles as the value clip range.
            if cfg.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -cfg.clip_coef,
                    cfg.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef
            if extra_loss_fn is not None:
                extra_loss = extra_loss_fn(mb_obs, mb_actions)
                loss = loss + extra_loss
                extra_loss_value = float(extra_loss.detach().item())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

        if cfg.target_kl is not None and approx_kl > cfg.target_kl:
            break

    y_pred = b_values.cpu().numpy()
    y_true = b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # Scalars come from the last minibatch processed, except clipfrac (mean over
    # all minibatches) and explained_variance (pre-update values).
    stats = {
        "value_loss": v_loss.item(),
        "policy_loss": pg_loss.item(),
        "entropy": entropy_loss.item(),
        "old_approx_kl": old_approx_kl.item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": float(np.mean(clipfracs)),
        "explained_variance": float(explained_var),
    }
    if extra_loss_value is not None:
        stats["extra_loss"] = extra_loss_value
    return stats
