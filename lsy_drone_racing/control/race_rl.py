"""Racing policy (PPO) for level 2 + deployment controller. Interface: ``control_mode = "attitude"``.

This file holds everything the trained policy needs, shared by training and deployment:

* ``features``: observation -> feature vector. Written once for numpy *and* jax (``xp``), so the
  training environment and the controller see exactly the same inputs.
* ``Agent``: actor-critic MLP with a built-in running observation normaliser.
* ``scale_action``: network output in [-1, 1] -> ``[roll, pitch, yaw, collective thrust]``.
* ``RaceRL``: the controller (loads ``race_rl.ckpt`` from this directory).

Train with ``python lsy_drone_racing/control/train_race_rl.py`` (needs the gpu pixi env), copy the
resulting ``race_rl.ckpt`` next to this file, set ``control_mode = "attitude"`` and
``controller.file = "race_rl.py"`` in ``level2.toml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from crazyflow.dynamics import load_params
from torch.distributions.normal import Normal

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

N_GATES = 4
N_POLES = 4
N_PASSES = 5
RP_MAX = 1.2  # rad, roll/pitch command at network output +-1
OBS_DIM = 59
CKPT = Path(__file__).parent / "race_rl.ckpt"


# --------------------------------------------------------------------------------------------------
# Features (numpy or jax)
# --------------------------------------------------------------------------------------------------
def quat_to_mat(q, xp):
    """Rotation matrices (..., 3, 3) from xyzw quaternions (..., 4)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = xp.sqrt(x * x + y * y + z * z + w * w) + 1e-9
    x, y, z, w = x / n, y / n, z / n, w / n
    r0 = xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1)
    r1 = xp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1)
    r2 = xp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1)
    return xp.stack([r0, r1, r2], axis=-2)


def _take(arr, idx, xp):
    """arr (B, N, ...) gathered with idx (B,) along axis 1 -> (B, ...)."""
    idx = idx.reshape((-1,) + (1,) * (arr.ndim - 1))
    return xp.take_along_axis(arr, idx, axis=1)[:, 0]


def features(obs: dict, last_action, xp):
    """Policy input (B, OBS_DIM). ``obs`` holds batched env observations, ``last_action`` is (B, 4)."""
    pos, vel, wb = obs["pos"], obs["vel"], obs["ang_vel"]
    rot = quat_to_mat(obs["quat"], xp)
    passed = obs["n_gates_passed"].astype(xp.int32)
    k1 = xp.minimum(passed, N_PASSES - 1)
    k2 = xp.minimum(passed + 1, N_PASSES - 1)
    seq, dirs = obs["gate_sequence"].astype(xp.int32), obs["gate_sequence_direction"]

    def gate_frame(k):
        gid = xp.take_along_axis(seq, k[:, None], axis=1)[:, 0]
        d = xp.take_along_axis(dirs, k[:, None], axis=1)[:, 0].astype(pos.dtype)
        gp = _take(obs["gates_pos"], gid, xp)
        gr = quat_to_mat(_take(obs["gates_quat"], gid, xp), xp)
        ex = gr[..., :, 0] * d[:, None]  # crossing direction
        ey, ez = gr[..., :, 1], gr[..., :, 2]
        vis = _take(obs["gates_visited"].astype(pos.dtype), gid, xp)
        return gp, ex, ey, ez, vis

    gp1, ex1, ey1, ez1, vis1 = gate_frame(k1)
    gp2, ex2, _, _, vis2 = gate_frame(k2)

    def local(v, ex, ey, ez):
        return xp.stack(
            [xp.sum(v * ex, -1), xp.sum(v * ey, -1), xp.sum(v * ez, -1)], axis=-1
        )

    rel1 = gp1 - pos
    p_loc = local(-rel1, ex1, ey1, ez1)  # drone in gate frame
    v_loc = local(vel, ex1, ey1, ez1)
    rel2 = gp2 - pos
    poles = (obs["obstacles_pos"][..., :2] - pos[:, None, :2]).reshape(pos.shape[0], -1)
    onehot = (k1[:, None] == xp.arange(N_PASSES)[None, :]).astype(pos.dtype)
    feats = [
        pos / 2.0,
        vel / 3.0,
        rot.reshape(pos.shape[0], 9),
        wb / 10.0,
        rel1 / 2.0,
        p_loc,
        v_loc / 3.0,
        ex1,
        vis1[:, None],
        rel2 / 2.0,
        ex2,
        vis2[:, None],
        poles / 2.0,
        obs["obstacles_visited"].astype(pos.dtype),
        onehot,
        last_action,
    ]
    return xp.concatenate(feats, axis=-1).astype(xp.float32)


def scale_action(a, t_min: float, t_max: float, xp):
    """Network output in [-1, 1]^4 -> [roll, pitch, yaw, thrust]; yaw is always commanded to 0."""
    a = xp.clip(a, -1.0, 1.0)
    roll, pitch = a[..., 0] * RP_MAX, a[..., 1] * RP_MAX
    thrust = 0.5 * (t_max + t_min) + a[..., 3] * 0.5 * (t_max - t_min)
    return xp.stack([roll, pitch, xp.zeros_like(roll), thrust], axis=-1)


# --------------------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------------------
def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    """Orthogonal initialisation."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """Actor-critic with running observation normalisation (stored in the checkpoint)."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = 4, hidden: int = 128):
        """Create the networks."""
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), -0.5))
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))
        self.register_buffer("obs_count", torch.tensor(1e-4))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise observations."""
        return torch.clamp((x - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -10.0, 10.0)

    @torch.no_grad()
    def update_norm(self, x: torch.Tensor) -> None:
        """Update the running mean/variance with a batch (B, obs_dim)."""
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), float(x.shape[0])
        delta = b_mean - self.obs_mean
        tot = self.obs_count + b_n
        self.obs_mean += delta * b_n / tot
        m2 = self.obs_var * self.obs_count + b_var * b_n + delta**2 * self.obs_count * b_n / tot
        self.obs_var.copy_(m2 / tot)
        self.obs_count.copy_(tot)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """State value."""
        return self.critic(self.norm(x))

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None, deterministic: bool = False
    ):
        """Sample (or evaluate) an action; returns action, logprob, entropy, value."""
        xn = self.norm(x)
        mean = self.actor_mean(xn)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(xn)


# --------------------------------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------------------------------
class RaceRL(Controller):
    """Runs the trained racing policy (attitude interface)."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Load the policy."""
        super().__init__(obs, info, config)
        params = load_params(config.sim.dynamics, config.sim.drone)
        self._t_min, self._t_max = float(params["thrust_min"]) * 4, float(params["thrust_max"]) * 4
        self._agent = Agent()
        self._agent.load_state_dict(torch.load(CKPT, map_location="cpu"))
        self._agent.eval()
        self._last = np.zeros((1, 4), dtype=np.float32)

    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray[np.floating]:
        """Return ``[roll, pitch, yaw, collective thrust]``."""
        batched = {k: np.asarray(v)[None] for k, v in obs.items()}
        batched["pos"] = batched["pos"].astype(np.float32)
        x = features(batched, self._last, np)
        with torch.no_grad():
            a, _, _, _ = self._agent.get_action_and_value(torch.from_numpy(x), deterministic=True)
        a = a.numpy().astype(np.float32)
        self._last = a.copy()
        return scale_action(a, self._t_min, self._t_max, np)[0].astype(np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        """Never finish early; the environment ends the episode after the last gate."""
        return False

    def episode_callback(self):
        """Reset the action memory."""
        self._last = np.zeros((1, 4), dtype=np.float32)
