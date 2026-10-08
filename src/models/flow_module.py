import torch
import torch.nn.functional as F
from typing import Any

from src.models.generic_module import GenericLitModule


class FlowMatchingModule(GenericLitModule):
    ""






























    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        name: str = "flow_unet",
        context_left: int = 30,
        context_right: int = 30,
        n_val_steps: int = 5,
        n_inference_steps: int = 20,
        vel_clip: float = 2.0,
        t_sample_mode: str = "uniform",   # "uniform" | "logit_normal"
        t_logit_mean: float = 0.0,        # used when t_sample_mode="logit_normal"
        t_logit_std: float = 1.0,
    ):
        # Properly initialise GenericLitModule to set up hparams (optimizer/scheduler)
        # and all val/test/plotting infrastructure. loss_type unused but required.
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            name=name,
            context_left=context_left,
            context_right=context_right,
            loss_type="mse",  # placeholder, not used
        )
        # Flow-specific params as plain attrs (cannot add to hparams after save_hyperparameters)
        self._n_val_steps = n_val_steps
        self._n_inference_steps = n_inference_steps
        self._vel_clip = vel_clip
        self._t_sample_mode = t_sample_mode
        self._t_logit_mean = t_logit_mean
        self._t_logit_std = t_logit_std

    # ------------------------------------------------------------------ #
    # Time step sampling                                                 #
    # ------------------------------------------------------------------ #

    def _sample_t(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sample flow time t ∈ (0,1) according to t_sample_mode.

        "uniform":
            t ~ U[0, 1]  (original Rectified Flow)
        "logit_normal":
            z ~ N(t_logit_mean, t_logit_std²),  t = sigmoid(z)
            Concentrates mass around t ≈ 0.5 where the velocity field is
            hardest to learn.  SD3 default: mean=0, std=1.
        """
        if self._t_sample_mode == "logit_normal":
            z = torch.randn(B, device=device, dtype=dtype)
            z = z * self._t_logit_std + self._t_logit_mean
            return torch.sigmoid(z)
        return torch.rand(B, device=device, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Core: velocity prediction + Euler ODE sampler                      #
    # ------------------------------------------------------------------ #

    def forward(self, calcium: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ""















        B, T, _ = calcium.shape
        t_channel = t.view(B, 1, 1).expand(B, T, 1)
        net_input = torch.cat([calcium, x_t, t_channel], dim=-1)
        return self.net(net_input)

    @staticmethod
    def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
        ""
        mn = x.min(dim=1, keepdim=True).values   # (B, 1, 1)
        mx = x.max(dim=1, keepdim=True).values
        return (x - mn) / (mx - mn + 1e-6)

    @torch.no_grad()
    def sample(self, calcium: torch.Tensor, n_steps: int) -> torch.Tensor:
        ""




        x = self._minmax_norm(calcium[:, :, 0:1].clone())
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = torch.full((x.shape[0],), float(i) * dt, device=calcium.device)
            v = self.forward(calcium, x, t_val)
            v = v.clamp(-self._vel_clip, self._vel_clip)
            v = torch.nan_to_num(v, nan=0.0)   
            x = x + v * dt
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        return x

    # ------------------------------------------------------------------ #
    # Override training_step: flow matching loss                         #
    # ------------------------------------------------------------------ #

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if len(batch) == 4:
            calcium, y_spike, _, _ = batch
        else:
            calcium, y_spike = batch[0], batch[1]

        B = calcium.shape[0]

        # Source x0 = per-window minmax(calcium) ∈ [0,1]
        
        x0 = self._minmax_norm(calcium[:, :, 0:1])  # (B, T, 1)

        # Sample random flow time t
        t = self._sample_t(B, calcium.device, calcium.dtype)

        # Rectified Flow linear interpolation
        t_view = t.view(B, 1, 1)
        x_t = (1.0 - t_view) * x0 + t_view * y_spike

        # Target velocity: "how to transform normalised calcium into spike"
        v_target = y_spike - x0

        v_pred = self.forward(calcium, x_t, t)
        loss = F.mse_loss(v_pred, v_target)

        self.log("loss of train", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------ #
    # Override model_step: used by parent validation/test pipeline       #
    # ------------------------------------------------------------------ #

    def model_step(self, batch: Any) -> tuple:
        meta = None
        if len(batch) == 4:
            calcium, y, w, meta = batch
        elif len(batch) == 3:
            calcium, y, w = batch
        else:
            calcium, y = batch[0], batch[1]
            w = None

        preds = self.sample(calcium, n_steps=self._n_val_steps)
        preds = preds.clamp(min=0.0)     # spike rate >= 0
        loss = F.mse_loss(preds, y)      # proxy loss for logging
        return loss, preds, y, w, meta


# ==========================================================================
# CondFlowMatchingModule — Gaussian source, calcium as conditioning
# ==========================================================================

class CondFlowMatchingModule(GenericLitModule):
    ""

































    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        name: str = "flow_unet_c",
        context_left: int = 30,
        context_right: int = 30,
        n_val_steps: int = 5,
        n_inference_steps: int = 20,
        vel_clip: float = 3.0,
        noise_std: float = 1.0,         # σ of source Gaussian x_0 ~ N(0, σ)
        use_sigmoid: bool = True,
        t_sample_mode: str = "uniform",  # "uniform" | "logit_normal"
        t_logit_mean: float = 0.0,       # used when t_sample_mode="logit_normal"
        t_logit_std: float = 1.0,
    ):
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            name=name,
            context_left=context_left,
            context_right=context_right,
            loss_type="mse",
        )
        self._n_val_steps      = n_val_steps
        self._n_inference_steps = n_inference_steps
        self._vel_clip         = vel_clip
        self._noise_std        = noise_std
        self._use_sigmoid      = use_sigmoid
        self._t_sample_mode    = t_sample_mode
        self._t_logit_mean     = t_logit_mean
        self._t_logit_std      = t_logit_std

    # ------------------------------------------------------------------ #
    # Time step sampling                                                  #
    # ------------------------------------------------------------------ #

    def _sample_t(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sample flow time t ∈ (0,1) — shared logic with FlowMatchingModule."""
        if self._t_sample_mode == "logit_normal":
            z = torch.randn(B, device=device, dtype=dtype)
            z = z * self._t_logit_std + self._t_logit_mean
            return torch.sigmoid(z)
        return torch.rand(B, device=device, dtype=dtype)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        calcium: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity v_θ(x_t, t | calcium).

        Args:
            calcium: (B, T, 1)   — conditioning calcium (z-scored)
            x_t    : (B, T, 1)   — noisy spike at flow time t
            t      : (B,)        — flow time in [0,1]
        Returns:
            velocity: (B, T, 1)  — dx/dt
        """
        B, T, _ = x_t.shape
        t_ch = t.view(B, 1, 1).expand(B, T, 1)          # (B, T, 1)
        net_input = torch.cat([x_t, t_ch], dim=-1)       # (B, T, 2)
        raw = self.net(net_input, calcium)               # CondSpikeNet
        if self._use_sigmoid:
            # Bound velocity to [-vel_clip, vel_clip] via sigmoid.
            return (torch.sigmoid(raw) - 0.5) * 2.0 * self._vel_clip
        return raw

    @torch.no_grad()
    def sample(self, calcium: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Euler ODE from x_0 ~ N(0, σ) conditioned on calcium."""
        x = torch.randn_like(calcium[:, :, 0:1]) * self._noise_std
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = torch.full((x.shape[0],), float(i) * dt, device=x.device)
            v = self.forward(calcium, x, t_val)
            v = v.clamp(-self._vel_clip, self._vel_clip)
            v = torch.nan_to_num(v, nan=0.0)
            x = x + v * dt
            x = x.clamp(-self._vel_clip * 5.0, self._vel_clip * 5.0)
        x = torch.nan_to_num(x, nan=0.0)
        return x

    # ------------------------------------------------------------------ #

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if len(batch) == 4:
            calcium, y_spike, _, _ = batch
        else:
            calcium, y_spike = batch[0], batch[1]

        B = calcium.shape[0]

        # Gaussian source
        x0 = torch.randn_like(y_spike) * self._noise_std  # (B, T, 1)

        # Flow time
        t = self._sample_t(B, calcium.device, calcium.dtype)
        t_view = t.view(B, 1, 1)

        # Linear interpolation: x_t = (1-t)*x0 + t*x1
        x_t = (1.0 - t_view) * x0 + t_view * y_spike

        # Target velocity
        v_target = y_spike - x0

        # Predicted velocity
        v_pred = self.forward(calcium, x_t, t)
        if self._use_sigmoid:
            v_target = v_target.clamp(-self._vel_clip, self._vel_clip)
        loss = F.mse_loss(v_pred, v_target)

        self.log("loss of train", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------ #

    def model_step(self, batch: Any) -> tuple:
        meta = None
        if len(batch) == 4:
            calcium, y, w, meta = batch
        elif len(batch) == 3:
            calcium, y, w = batch
        else:
            calcium, y = batch[0], batch[1]
            w = None

        preds = self.sample(calcium, n_steps=self._n_val_steps)
        preds = preds.clamp(min=0.0)
        loss = F.mse_loss(preds, y)
        return loss, preds, y, w, meta
