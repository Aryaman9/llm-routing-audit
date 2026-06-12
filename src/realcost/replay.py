"""Replay engine: run a routing policy over a conversation/trajectory and
account its cost under CM0 and CM1 simultaneously.

A Trajectory is provider-agnostic token arithmetic:
  turn t prompt length = system + sum over s<t of (user_s + output_s) + user_t
Output lengths can vary by chosen model (response-matrix mode, Phase 3) or be
fixed from logs (frozen-history mode, Phase 0 / E0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .cache_sim import PrefixCacheSim
from .cost import CostBreakdown, accumulate
from .pricing import CacheParams, ModelPrice


@dataclass
class Turn:
    user_tokens: int
    t_seconds: float                       # timestamp since conversation start
    output_tokens: int | dict[str, int]    # int (frozen) or per-model dict (matrix)

    def out_for(self, model: str) -> int:
        if isinstance(self.output_tokens, dict):
            return self.output_tokens[model]
        return self.output_tokens


@dataclass
class Trajectory:
    turns: list[Turn]
    system_tokens: int = 0
    conv_id: str = ""


# A policy maps (turn_index, trajectory) -> model name. Stateless wrappers for
# stateful policies (e.g., learned routers, hysteresis) close over their state.
Policy = Callable[[int, Trajectory], str]


def fixed(model: str) -> Policy:
    return lambda i, traj: model


def alternate(models: Sequence[str], period: int = 1) -> Policy:
    return lambda i, traj: models[(i // period) % len(models)]


def switch_at(first: str, second: str, switch_turn: int) -> Policy:
    return lambda i, traj: first if i < switch_turn else second


@dataclass
class ReplayResult:
    breakdown: CostBreakdown
    decisions: list[str] = field(default_factory=list)


def replay(
    traj: Trajectory,
    policy: Policy,
    prices: dict[str, ModelPrice],
    cache_params: dict[str, CacheParams],
) -> ReplayResult:
    sim = PrefixCacheSim(cache_params)
    bd = CostBreakdown()
    decisions: list[str] = []
    prompt_len = traj.system_tokens
    prev_model: str | None = None

    for i, turn in enumerate(traj.turns):
        model = policy(i, traj)
        decisions.append(model)
        prompt_len += turn.user_tokens
        out = turn.out_for(model)
        acc = sim.account_call(model, prompt_len, out, now=turn.t_seconds)
        accumulate(bd, acc, prices[model], cache_params[model],
                   switched=(prev_model is not None and model != prev_model))
        prompt_len += out  # assistant reply joins the next turn's prompt
        prev_model = model

    return ReplayResult(breakdown=bd, decisions=decisions)
