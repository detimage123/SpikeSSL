import os
import json
import inspect
import numpy as np
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.regression import PearsonCorrCoef
from src.utils.signal_processing import smooth_spikes

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GenericLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        name: str = None,
        context_left: int = 30,
        context_right: int = 30,
        loss_type: str = "mse",
        loss_alpha: float = 20.0,   
        loss_lambda: float = 0.5,   
        loss_beta: float = 2.0,     
        loss_beta_under: float = 0.3,  
        loss_delta: float = 0.1,    # huber delta
        loss_mean_lambda: float = 0.2,  
        loss_peak_q: float = 0.92,  
        loss_peak_alpha: float = 1.0,  
        loss_zero_thr: float = 0.02,    
        # gated sparse output
        gate_loss_weight: float = 1.0,    # BCE weight for gate detection head
        gate_focal_gamma: float = 0.0,    # >0 enables focal BCE for gate (NOT recommended, kept for compat)
        gate_bg_penalty: float = 0.0,     # >0 adds direct L1 suppression on gate in background frames
                                          # gate_loss = weight*BCE + bg_penalty*mean(gate[target==0])
                                          # different from focal: targets gate *activations* not gradient shape
                                          # recommended range: 0.05 ~ 0.3
        # direct correlation optimization
        loss_corr_lambda: float = 0.0,    # CCC penalty weight; >0 enables concordance corr push
        # few-shot adaptation
        pretrained_ckpt: str = "",         # path to pretrained .ckpt (LOO checkpoint)
        freeze_mode: str = "none",         # none | prompt_only
        # smoothing: float = 0.1,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])
        
        self.net = net
        self._name = name if name is not None else self.__class__.__name__
        
        self.test_corr = PearsonCorrCoef()
        self.val_corr = PearsonCorrCoef()
        self.test_data = []
        self.val_data = []
        self.context_left = context_left
        self.context_right = context_right
        self.best_val_corr = float("-inf")
        self.best_val_epoch = 0
        self.best_val_metric = float("-inf")  # tracks best (corr - |bias|) for global best record

        # Check which optional kwargs net.forward accepts (for SpikeSSM-MM variants)
        _fwd_params = inspect.signature(self.net.forward).parameters
        self._net_accepts_indicator = 'indicator_id' in _fwd_params
        self._net_accepts_fs = 'fs' in _fwd_params
        self._net_accepts_species = 'species_id' in _fwd_params
        self._net_accepts_tissue = 'tissue_id' in _fwd_params
        self._net_accepts_cell_type = 'cell_type_id' in _fwd_params

        # Load pretrained checkpoint + apply freeze
        if pretrained_ckpt:
            self._load_pretrained_and_freeze(pretrained_ckpt, freeze_mode)

    # ----------------------- Pretrained ckpt + freeze ----------------------- #

    def _load_pretrained_and_freeze(self, ckpt_path: str, freeze_mode: str):
        """Load pretrained weights from .ckpt and freeze parameters per freeze_mode."""
        import lightning as L

        print(f"\n[Few-shot] Loading pretrained checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Filter out target_prompt from pretrained (we want fresh init)
        state_dict = {k: v for k, v in state_dict.items()
                      if "target_prompt" not in k}

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (expected for new params): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")

        # ---- Freeze all first, then selectively unfreeze ----
        if freeze_mode == "none":
            print(f"[Few-shot] freeze_mode=none: all params trainable")
            return

        for name, param in self.named_parameters():
            param.requires_grad = False

        def _unfreeze_output_heads(net):
            """Unfreeze head + gate_head + var_proj + indicator_var_bias."""
            for p in net.head.parameters():
                p.requires_grad = True
            if hasattr(net, 'gate_head'):
                for p in net.gate_head.parameters():
                    p.requires_grad = True
            if hasattr(net, 'var_proj'):
                for p in net.var_proj.parameters():
                    p.requires_grad = True
            if hasattr(net, 'indicator_var_bias'):
                for p in net.indicator_var_bias.parameters():
                    p.requires_grad = True

        if freeze_mode == "embed_only":
            # Unfreeze: only indicator_embed (960 params)
            # Safest mode: the target indicator's embedding row was
            # untrained during LOO, so it can only improve. Output
            # heads stay at their pretrained calibration.
            for p in self.net.cond_encoder.indicator_embed.parameters():
                p.requires_grad = True

        elif freeze_mode == "embed_head":
            # Unfreeze: indicator_embed + output heads (~1.5K params)
            for p in self.net.cond_encoder.indicator_embed.parameters():
                p.requires_grad = True
            _unfreeze_output_heads(self.net)

        elif freeze_mode == "embed_cond_head":
            # Unfreeze: indicator_embed + fuse + output heads (~18K params)
            for p in self.net.cond_encoder.indicator_embed.parameters():
                p.requires_grad = True
            for p in self.net.cond_encoder.fuse.parameters():
                p.requires_grad = True
            _unfreeze_output_heads(self.net)

        else:
            raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Few-shot] freeze_mode={freeze_mode}: {n_train} trainable params")

        print(f"[Few-shot] Pretrained checkpoint loaded successfully.\n")

    # ------------------------- Global best record ------------------------- #

    @staticmethod
    def _sanitize_for_json(x):
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if isinstance(x, dict):
            return {str(k): GenericLitModule._sanitize_for_json(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [GenericLitModule._sanitize_for_json(v) for v in x]
        try:
            import numpy as _np
            if isinstance(x, (_np.floating, _np.integer)):
                return x.item()
        except Exception:
            pass
        return str(x)

    def _global_best_file_path(self) -> str:
        try:
            base_dir = hydra.utils.get_original_cwd()
        except Exception:
            base_dir = os.getcwd()

        best_dir_override = os.environ.get("SPIKESSM_BEST_CONFIG_DIR", "").strip()

        
        # model=spikessm_mm_v2_var + data=loo_ogb1 → global_best_config_mm_v2_var_loo_ogb1.json
        # model=spikessm_mm_v2_var + default data  → global_best_config_mm_v2_var.json
        # model=spikessm_mm_v2     + default data  → global_best_config_mm_v2.json
        
        suffix = ""
        try:
            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra.instance().is_initialized():
                from hydra.core.hydra_config import HydraConfig
                choices = HydraConfig.get().runtime.choices
                # model suffix: strip "spikessm_" prefix to keep names short
                model_name = choices.get("model", None)
                if model_name and model_name != "null":
                    model_suffix = model_name.replace("spikessm_", "")
                    suffix += f"_{model_suffix}"
                # data suffix
                data_name = choices.get("data", None)
                if data_name and data_name not in ("null", "calcium"):
                    suffix += f"_{data_name}"
        except Exception:
            pass

        # Few-shot results go to a separate file (triggered by pretrained_ckpt)
        pretrained_ckpt = getattr(self.hparams, "pretrained_ckpt", "")
        if pretrained_ckpt:
            suffix += "_fewshot"

        if best_dir_override:
            target_dir = best_dir_override
            if not os.path.isabs(target_dir):
                target_dir = os.path.join(base_dir, target_dir)
            return os.path.join(target_dir, f"global_best_config{suffix}.json")

        return os.path.join(base_dir, "result", f"global_best_config{suffix}.json")

    def _read_global_best_record(self) -> Dict[str, Any]:
        path = self._global_best_file_path()
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            init_record = {
                "best_corr": 0.0,
                "updated_at": None,
                "best_run": None,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(init_record, f, indent=2, ensure_ascii=False)
            return init_record

        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            if "best_corr" not in rec:
                rec["best_corr"] = 0.0
            return rec
        except Exception:
            return {
                "best_corr": 0.0,
                "updated_at": None,
                "best_run": None,
            }

    @staticmethod
    def _pearson_corr_np(x: np.ndarray, y: np.ndarray) -> float:
        """Pearson correlation using numpy float64 — numerically stable on long vectors."""
        x64 = x.astype(np.float64)
        y64 = y.astype(np.float64)
        vx = x64 - x64.mean()
        vy = y64 - y64.mean()
        denom = np.sqrt(np.sum(vx ** 2) * np.sum(vy ** 2))
        if denom < 1e-12:
            return 0.0
        return float(np.sum(vx * vy) / denom)

    def _maybe_update_global_best_record(self, corr_float: float, error: float, bias: float,
                                           pred_stats: dict = None, target_stats: dict = None) -> None:
        rec = self._read_global_best_record()
        # Criterion: maximise (corr - 0.01*|bias|).
        # Bias has much lower weight (0.01) compared to corr, so corr dominates.
        # Example: corr=0.90, bias=0.05 → score = 0.90 - 0.0005 = 0.8995
        #          corr=0.88, bias=0.01 → score = 0.88 - 0.0001 = 0.8799 (first is better)
        score = corr_float - 0.01 * abs(bias)
        if "best_metric" in rec:
            prev_best_metric = float(rec["best_metric"])
        else:
            # Fallback for old-format records written before best_metric was introduced:
            # reconstruct the score from stored corr and bias (accurate, not a rough offset).
            old_corr = float(rec.get("best_corr", 0.0))
            old_bias = float(rec.get("best_bias", 0.0))
            prev_best_metric = old_corr - 0.01 * abs(old_bias)
        if score <= prev_best_metric:
            return

        path = self._global_best_file_path()
        optim_lr = None
        try:
            optim_lr = float(self.optimizers().param_groups[0]["lr"])
        except Exception:
            optim_lr = None

        loss_type = str(getattr(self.hparams, "loss_type", "unknown"))

        # Collect model architecture parameters from net
        net_params = {}
        if hasattr(self, "net"):
            net = self.net
            # Grab all __init__ args stored in the net (common pattern)
            for attr in sorted(vars(net)):
                if attr.startswith("_"):
                    continue
                val = getattr(net, attr)
                if isinstance(val, (int, float, bool, str)):
                    net_params[attr] = val
            # Also try to get from constructor signature defaults
            import inspect
            try:
                sig = inspect.signature(net.__class__.__init__)
                for name, param in sig.parameters.items():
                    if name == "self":
                        continue
                    if name not in net_params and hasattr(net, name):
                        val = getattr(net, name)
                        if isinstance(val, (int, float, bool, str)):
                            net_params[name] = val
            except (ValueError, TypeError):
                pass

        run_info = {
            "model_name": self._name,
            "module_class": self.__class__.__name__,
            "net_class": self.net.__class__.__name__ if hasattr(self, "net") else None,
            "epoch": int(self.current_epoch + 1),
            "loss_type": loss_type,
            "corr": float(corr_float),
            "error": float(error),
            "bias": float(bias),
            "lr": optim_lr,
            "num_params": int(sum(p.numel() for p in self.parameters())),
            "net_config": net_params,
            "hparams": self._sanitize_for_json(dict(self.hparams)),
        }

        from datetime import datetime
        new_rec = {
            "best_metric": float(score),   # corr - 0.01*|bias|: the actual optimisation criterion
            "best_corr": float(corr_float),
            "best_error": float(error),
            "best_bias": float(bias),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "best_run": run_info,
        }
        if pred_stats:
            new_rec["pred_stats"] = pred_stats
        if target_stats:
            new_rec["target_stats"] = target_stats

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(new_rec, f, indent=2, ensure_ascii=False)
        print(
            f"  -> global best updated: metric(corr-0.01*|bias|) {prev_best_metric:.4f} -> {score:.4f} "
            f"(corr={corr_float:.4f}, bias={bias:.4f})  loss_type={loss_type}  ({path})"
        )


    def _build_net_kwargs(self, meta: dict = None) -> dict:
        """Build kwargs dict for self.net.forward based on detected signature."""
        kwargs = {}
        if meta is None:
            return kwargs
        if self._net_accepts_indicator:
            kwargs['indicator_id'] = meta.get('indicator_id')
        if self._net_accepts_fs:
            kwargs['fs'] = meta.get('sampling_rate')
        if self._net_accepts_species:
            kwargs['species_id'] = meta.get('species_id')
        if self._net_accepts_tissue:
            kwargs['tissue_id'] = meta.get('tissue_id')
        if self._net_accepts_cell_type:
            kwargs['cell_type_id'] = meta.get('cell_type_id')
        return kwargs

    def forward(
        self,
        x: torch.Tensor,
        indicator_id: torch.Tensor = None,
        fs: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        net_kw = {}
        if self._net_accepts_indicator and indicator_id is not None:
            net_kw['indicator_id'] = indicator_id
        if self._net_accepts_fs and fs is not None:
            net_kw['fs'] = fs
        if self._net_accepts_species and 'species_id' in kwargs:
            net_kw['species_id'] = kwargs['species_id']
        if self._net_accepts_tissue and 'tissue_id' in kwargs:
            net_kw['tissue_id'] = kwargs['tissue_id']
        if self._net_accepts_cell_type and 'cell_type_id' in kwargs:
            net_kw['cell_type_id'] = kwargs['cell_type_id']
        return self.net(x, **net_kw)


    @staticmethod
    def _weighted_reduce(loss: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        ""




        if weights is None:
            return loss.mean()
        w = weights.to(loss.dtype)
        denom = w.sum().clamp(min=1.0)
        return (loss * w).sum() / denom

    def _batch_pearson_loss(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        ""





        p = preds.reshape(-1)
        t = targets.reshape(-1)
        if weights is not None:
            mask = weights.reshape(-1) > 0
            p = p[mask]
            t = t[mask]
        vx = p - p.mean()
        vy = t - t.mean()
        corr = (vx * vy).sum() / (
            (vx.pow(2).sum() + 1e-8).sqrt() * (vy.pow(2).sum() + 1e-8).sqrt()
        )
        corr = torch.nan_to_num(corr, nan=0.0).clamp(-1.0, 1.0)
        return 1.0 - corr

    def _mean_bias_penalty(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
        """Penalty on global mean mismatch to reduce systematic over-prediction."""
        if weights is None:
            p_mean = preds.mean()
            t_mean = targets.mean()
        else:
            w = weights.to(preds.dtype)
            denom = w.sum().clamp(min=1.0)
            p_mean = (preds * w).sum() / denom
            t_mean = (targets * w).sum() / denom
        return (p_mean - t_mean).pow(2)

    def _over_pred_penalty(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
        """Asymmetric penalty: penalize over-estimation harder than under-estimation."""
        over = F.relu(preds - targets)
        return self._weighted_reduce(over.pow(2), weights)

    def _peak_under_penalty(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
        ""




        q = float(self.hparams.loss_peak_q)
        peak_alpha = float(self.hparams.loss_peak_alpha)

        tgt = targets.detach().clamp(min=0.0)
        if weights is not None:
            valid = (weights > 0)
            tgt_valid = tgt[valid]
            if tgt_valid.numel() == 0:
                return torch.tensor(0.0, device=targets.device, dtype=targets.dtype)
            thr = torch.quantile(tgt_valid, q)
            peak_mask = (tgt >= thr).to(tgt.dtype) * weights.to(tgt.dtype)
            t_scale = torch.quantile(tgt_valid, 0.99).clamp(min=1e-3)
        else:
            thr = torch.quantile(tgt.reshape(-1), q)
            peak_mask = (tgt >= thr).to(tgt.dtype)
            t_scale = torch.quantile(tgt.reshape(-1), 0.99).clamp(min=1e-3)

        t_norm = (tgt / t_scale).clamp(0.0, 1.0)
        under = F.relu(targets - preds)
        # use huber on positive under-error to avoid overly strong pull on peaks
        under_huber = F.huber_loss(under, torch.zeros_like(under), reduction="none", delta=self.hparams.loss_delta)
        under = under_huber * (1.0 + peak_alpha * t_norm) * peak_mask

        denom = peak_mask.sum().clamp(min=1.0)
        return under.sum() / denom

    def _gate_detection_loss(self, targets: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
        ""














        gate = getattr(self.net, '_last_gate', None)
        if gate is None:
            return torch.tensor(0.0, device=targets.device, dtype=targets.dtype)

        thr = float(self.hparams.loss_zero_thr)
        gamma_neg = float(self.hparams.gate_focal_gamma)
        bg_pen = float(self.hparams.gate_bg_penalty)
        gate_target = (targets > thr).float()   # (B, T, 1)
        gate_safe = torch.nan_to_num(gate, nan=0.5).clamp(1e-7, 1.0 - 1e-7)

        # ── Standard / Focal BCE ──
        bce = F.binary_cross_entropy(gate_safe, gate_target, reduction='none')
        if gamma_neg > 0:
            # Down-weight easy negatives: gate^gamma for GT=0 positions (kept for compat)
            asym_weight = gate_target + (1.0 - gate_target) * gate_safe.pow(gamma_neg)
            bce = asym_weight * bce
        bce_val = self._weighted_reduce(bce, weights)

        # ── Background Gate Penalty ──
        # Directly suppresses gate activations in zero-target frames.
        # Unlike focal, this is an explicit value-based regularizer, not a gradient re-weighting.
        if bg_pen > 0.0:
            bg_mask = (gate_target < 0.5)  # frames where GT binary = 0
            if weights is not None:
                w_flat = weights.to(gate_safe.dtype)
                bg_mask = bg_mask & (w_flat > 0)
            if bg_mask.any():
                bg_penalty = gate_safe[bg_mask].mean()
            else:
                bg_penalty = gate_safe.new_zeros(())
            return bce_val + bg_pen * bg_penalty

        return bce_val

    def _masked_mse_loss(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None, return_components: bool = False):
        mse = (preds - targets).pow(2)
        base = self._weighted_reduce(mse, weights)
        
        if return_components:
            z = torch.zeros_like(base)
            return base, {
                "base": base,
                "corr_pen": z,
                "over_pen": z,
                "under_peak": z,
                "zero_fp": z,
                "mean_pen": z,
                "l2_reg": z,
            }
        return base

    def _weighted_mse_loss(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None, return_components: bool = False):
        ""




        alpha = self.hparams.loss_alpha
        beta = self.hparams.loss_beta

        t_pos = targets.detach().clamp(min=0.0)
        if weights is not None:
            valid = (weights > 0).detach()
            if valid.any():
                t_scale = torch.quantile(t_pos[valid], 0.99).clamp(min=1e-3)
            else:
                t_scale = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)
        else:
            t_scale = torch.quantile(t_pos.reshape(-1), 0.99).clamp(min=1e-3)

        t_norm = (t_pos / t_scale).clamp(0.0, 1.0)
        spike_w = 1.0 + alpha * t_norm

        mse = (preds - targets).pow(2) * spike_w
        base = self._weighted_reduce(mse, weights)
        over_pen = self._over_pred_penalty(preds, targets, weights)
        mean_pen = self._mean_bias_penalty(preds, targets, weights)
        total = base + beta * over_pen + self.hparams.loss_mean_lambda * mean_pen
        if return_components:
            z = torch.zeros_like(base)
            return total, {
                "base": base,
                "corr_pen": z,
                "over_pen": over_pen,
                "under_peak": z,
                "zero_fp": z,
                "mean_pen": mean_pen,
                "l2_reg": z,
            }
        return total

    def _corr_mse_loss(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None, return_components: bool = False):
        ""



        lam = self.hparams.loss_lambda
        beta = self.hparams.loss_beta

        mse = self._weighted_reduce((preds - targets).pow(2), weights)
        corr = self._weighted_pearson(preds, targets, weights)
        corr_pen = lam * (1.0 - corr)
        over_pen = self._over_pred_penalty(preds, targets, weights)
        mean_pen = self._mean_bias_penalty(preds, targets, weights)
        total = mse + corr_pen + beta * over_pen + self.hparams.loss_mean_lambda * mean_pen
        if return_components:
            z = torch.zeros_like(mse)
            return total, {
                "base": mse,
                "corr_pen": corr_pen,
                "over_pen": over_pen,
                "under_peak": z,
                "zero_fp": z,
                "mean_pen": mean_pen,
                "l2_reg": z,
            }
        return total

    def _poisson_loss(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None, return_components: bool = False):
        ""




        lam = self.hparams.loss_lambda
        beta = self.hparams.loss_beta

        rate = preds.clamp(min=1e-6)
        tgt = targets.clamp(min=0.0)
        nll = rate - tgt * torch.log(rate)
        nll = self._weighted_reduce(nll, weights)

        l2_reg = self._weighted_reduce(rate.pow(2), weights)
        over_pen = self._over_pred_penalty(rate, tgt, weights)
        mean_pen = self._mean_bias_penalty(rate, tgt, weights)
        total = nll + lam * l2_reg + beta * over_pen + self.hparams.loss_mean_lambda * mean_pen
        if return_components:
            z = torch.zeros_like(nll)
            return total, {
                "base": nll,
                "corr_pen": z,
                "over_pen": over_pen,
                "under_peak": z,
                "zero_fp": z,
                "mean_pen": mean_pen,
                "l2_reg": l2_reg,
            }
        return total

    def _asym_huber_loss(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None, return_components: bool = False):
        ""



        delta = self.hparams.loss_delta
        beta = self.hparams.loss_beta
        huber = F.huber_loss(preds, targets, reduction="none", delta=delta)
        huber = self._weighted_reduce(huber, weights)
        over_pen = self._over_pred_penalty(preds, targets, weights)
        mean_pen = self._mean_bias_penalty(preds, targets, weights)
        total = huber + beta * over_pen + self.hparams.loss_mean_lambda * mean_pen
        if return_components:
            z = torch.zeros_like(huber)
            return total, {
                "base": huber,
                "corr_pen": z,
                "over_pen": over_pen,
                "under_peak": z,
                "zero_fp": z,
                "mean_pen": mean_pen,
                "l2_reg": z,
            }
        return total

    def _huber_loss(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None, return_components: bool = False):
        """Pure masked Huber loss without extra regularization terms."""
        delta = self.hparams.loss_delta
        huber = F.huber_loss(preds, targets, reduction="none", delta=delta)
        base = self._weighted_reduce(huber, weights)
        if return_components:
            z = torch.zeros_like(base)
            return base, {
                "base": base,
                "corr_pen": z,
                "over_pen": z,
                "under_peak": z,
                "zero_fp": z,
                "mean_pen": z,
                "l2_reg": z,
            }
        return base

    def _bys_peak_huber_loss(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None, return_components: bool = False):
        """Heteroscedastic peak-Huber loss with Bayesian aleatoric variance.

        Attenuation loss (Kendall & Gal, NeurIPS 2017):
          L_bys = E[ exp(-s) · huber(μ, y) + 0.5 · s ]   (masked)

        Where s = log σ²_t is retrieved from self.net._last_log_var.

        Intuition:
          - High-confidence frames (s small, σ² ≈ 0): exp(-s) ≈ 1/σ²→large,
            gradient on μ is amplified — model learns from these aggressively.
          - Noisy/unknown-indicator frames (s large, σ²→big): exp(-s)→0,
            Huber gradient is attenuated — model predicts conservatively,
            naturally suppressing false positives in low-SNR trials.
          - 0.5·s prevents the degenerate solution σ²→∞.

        Asymmetric penalties (over_pen, peak_under, mean_pen) are NOT
        attenuated by precision — they are hard structural constraints that
        must hold regardless of predicted uncertainty.

        Falls back to _peak_huber_loss if model has no _last_log_var.
        """
        log_var = getattr(self.net, '_last_log_var', None)
        if log_var is None:
            # Graceful fallback: model doesn't support Bayesian uncertainty
            return self._peak_huber_loss(preds, targets, weights, return_components)

        delta = self.hparams.loss_delta
        beta_over = self.hparams.loss_beta
        beta_under = self.hparams.loss_beta_under
        corr_lam = float(self.hparams.loss_corr_lambda)

        # s = log σ²; already clamped in model.forward
        s = log_var                                         # (B, T, 1)
        prec = torch.exp(-s)                               # 1/σ²; (B, T, 1)

        # ── Attenuated Huber (main heteroscedastic term) ──────────────── #
        huber_raw = F.huber_loss(
            preds, targets, reduction="none", delta=delta
        )                                                   # (B, T, 1)
        atten_huber = prec * huber_raw + 0.5 * s           # (B, T, 1)
        base = self._weighted_reduce(atten_huber, weights)

        # ── Structural penalties (NOT precision-weighted) ──────────────── #
        # Over-prediction: hard asymmetric constraint regardless of σ²
        over_pen = self._over_pred_penalty(preds, targets, weights)

        # Peak under-prediction: pull up missed peaks regardless of σ²
        under_peak = self._peak_under_penalty(preds, targets, weights)

        # Mean bias: prevent systematic offset
        mean_pen = self._mean_bias_penalty(preds, targets, weights)

        # Optional batch Pearson
        if corr_lam > 0:
            corr_pen = corr_lam * self._batch_pearson_loss(preds, targets, weights)
        else:
            corr_pen = torch.zeros_like(base)

        total = (
            base
            + beta_over * over_pen
            + beta_under * under_peak
            + self.hparams.loss_mean_lambda * mean_pen
            + corr_pen
        )

        if return_components:
            z = torch.zeros_like(base)
            return total, {
                "base": base,          # attenuated Huber (includes 0.5·s entropy)
                "corr_pen": corr_pen,
                "over_pen": over_pen,
                "under_peak": under_peak,
                "zero_fp": z,
                "mean_pen": mean_pen,
                "l2_reg": z,
            }
        return total

    def _peak_huber_loss(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None, return_components: bool = False):
        ""




        delta = self.hparams.loss_delta
        beta_over = self.hparams.loss_beta
        beta_under = self.hparams.loss_beta_under
        corr_lam = float(self.hparams.loss_corr_lambda)

        huber = F.huber_loss(preds, targets, reduction="none", delta=delta)
        huber = self._weighted_reduce(huber, weights)
        over_pen = self._over_pred_penalty(preds, targets, weights)
        under_peak = self._peak_under_penalty(preds, targets, weights)
        mean_pen = self._mean_bias_penalty(preds, targets, weights)

        total = (
            huber
            + beta_over * over_pen
            + beta_under * under_peak
            + self.hparams.loss_mean_lambda * mean_pen
        )

        if corr_lam > 0:
            pearson_loss = self._batch_pearson_loss(preds, targets, weights)
            corr_pen = corr_lam * pearson_loss
            total = total + corr_pen
        else:
            corr_pen = torch.zeros_like(huber)

        if return_components:
            z = torch.zeros_like(huber)
            return total, {
                "base": huber,
                "corr_pen": corr_pen,
                "over_pen": over_pen,
                "under_peak": under_peak,
                "zero_fp": z,
                "mean_pen": mean_pen,
                "l2_reg": z,
            }
        return total

    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None, return_components: bool = False):
        t = self.hparams.loss_type
        if t == "asym_huber":
            result = self._asym_huber_loss(preds, targets, weights, return_components)
        elif t == "huber":
            result = self._huber_loss(preds, targets, weights, return_components)
        elif t == "peak_huber":
            result = self._peak_huber_loss(preds, targets, weights, return_components)
        elif t == "bys_peak_huber":
            result = self._bys_peak_huber_loss(preds, targets, weights, return_components)
        elif t == "mse":
            result = self._masked_mse_loss(preds, targets, weights, return_components)
        else:
            raise ValueError(
                f"Unsupported loss_type='{t}'. Supported: mse, huber, asym_huber, peak_huber, bys_peak_huber"
            )

        # Gated sparse output: add BCE on spike gate if the net provides one
        gate = getattr(self.net, '_last_gate', None)
        if gate is not None:
            gate_bce = self._gate_detection_loss(targets, weights)
            gate_w = float(self.hparams.gate_loss_weight)
            if return_components:
                total, comp = result
                total = total + gate_w * gate_bce
                comp["gate_bce"] = gate_bce
                return total, comp
            else:
                return result + gate_w * gate_bce

        if return_components:
            total, comp = result
            comp["gate_bce"] = torch.zeros_like(total)
            return total, comp
        return result


    def model_step(self, batch: Any) -> tuple:
        meta = None
        if len(batch) == 4:
            x, y, w, meta = batch
        elif len(batch) == 3:
            x, y, w = batch
        else:
            x, y = batch
            w = None

        net_kw = self._build_net_kwargs(meta)
        preds = self.net(x, **net_kw)
        loss = self.compute_loss(preds, y, w)
        return loss, preds, y, w, meta


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if len(batch) == 4:
            x, y, w, meta = batch
        elif len(batch) == 3:
            x, y, w = batch
            meta = None
        else:
            x, y = batch
            w = None
            meta = None

        # Propagate epoch to net (used by v2_var variance warmup)
        if hasattr(self.net, '_current_epoch'):
            self.net._current_epoch = self.current_epoch

        net_kw = self._build_net_kwargs(meta)
        preds = self.net(x, **net_kw)
        loss, comp = self.compute_loss(preds, y, w, return_components=True)
        self.log("loss of train", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_base", comp["base"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_corr_pen", comp["corr_pen"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_over_pen", comp["over_pen"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_under_peak", comp["under_peak"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_zero_fp", comp["zero_fp"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_mean_pen", comp["mean_pen"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_l2_reg", comp["l2_reg"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_gate_bce", comp["gate_bce"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        epoch = self.current_epoch
        train_loss = self.trainer.callback_metrics.get("loss of train_epoch", None)
        loss_type = self.hparams.loss_type
        lr = self.optimizers().param_groups[0]["lr"]

        def _v(key, default=0.0):
            v = self.trainer.callback_metrics.get(key, None)
            if v is None:
                return float(default)
            try:
                return float(v)
            except Exception:
                return float(default)

        base_v = _v("train_base")
        corr_pen_v = _v("train_corr_pen")
        over_pen_v = _v("train_over_pen")
        under_peak_v = _v("train_under_peak")
        mean_pen_v = _v("train_mean_pen")
        gate_bce_v = _v("train_gate_bce")

        beta = float(self.hparams.loss_beta)
        beta_under = float(self.hparams.loss_beta_under)
        mean_lam = float(self.hparams.loss_mean_lambda)
        gate_w = float(self.hparams.gate_loss_weight)

        # Build compact parts list: name=weighted_value (only non-zero terms)
        corr_lam = float(self.hparams.loss_corr_lambda)
        parts = [f"base={base_v:.4f}"]
        if loss_type in {"asym_huber", "peak_huber"} and over_pen_v > 1e-8:
            parts.append(f"over={beta * over_pen_v:.4f}")
        if loss_type == "peak_huber" and under_peak_v > 1e-8:
            parts.append(f"under={beta_under * under_peak_v:.4f}")
        if loss_type in {"asym_huber", "peak_huber"} and mean_pen_v > 1e-8:
            parts.append(f"mean={mean_lam * mean_pen_v:.4f}")
        if loss_type == "peak_huber" and corr_lam > 0 and corr_pen_v > 1e-8:
            parts.append(f"corr={corr_pen_v:.4f}")
        if gate_bce_v > 1e-8:
            parts.append(f"gate={gate_w * gate_bce_v:.4f}")

        loss_str = f"{float(train_loss):.4f}" if train_loss is not None else "N/A"
        print(f"[Epoch {epoch+1}] loss={loss_str} lr={lr:.2e} ({loss_type})  {' | '.join(parts)}")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss, preds, targets, weight, meta = self.model_step(batch)
        self.log("loss of val", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        x = batch[0]
        entry = {
            "inputs": x.detach().cpu().numpy(),
            "preds": preds.detach().cpu().numpy(),
            "targets": targets.detach().cpu().numpy(),
            "meta": meta,
        }
        # Store per-frame log-variance if the model produces it (v2_var)
        log_var = getattr(self.net, '_last_log_var', None)
        if log_var is not None:
            entry["log_var"] = log_var.detach().cpu().numpy()
        self.val_data.append(entry)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self.val_data.clear()
            return

        epoch = self.current_epoch  # 0-indexed

        if len(self.val_data) == 0:
            return

        all_trial_preds, all_trial_targets, _ = self._trial_reconstruction(data=self.val_data)
        all_trial_inputs, _, _ = self._trial_reconstruction(data=self.val_data, use_input=True)

        # Reconstruct per-trial variance if available (v2_var model)
        has_var = all("log_var" in d for d in self.val_data)
        if has_var:
            # Average variance in LINEAR space (not log space) to respect Jensen's inequality.
            # _trial_reconstruction averages values per-frame; we feed exp(log_var) = var.
            all_trial_var, _, _ = self._trial_reconstruction(
                data=self.val_data, field="log_var", pre_transform=lambda x: np.exp(np.clip(x, -10, 10))
            )
            # var (linear-averaged) → std
            all_trial_std = [np.sqrt(np.clip(v, 0, None)) for v in all_trial_var]
        else:
            all_trial_std = None

        all_preds_cat = np.clip(np.concatenate(all_trial_preds), 0, None)
        all_targets_cat = np.concatenate(all_trial_targets)
        all_preds_raw = np.concatenate(all_trial_preds)

        
        corr_float = self._pearson_corr_np(all_preds_raw, all_targets_cat)
        self.log("val_corr", corr_float, prog_bar=True)

        diff = all_preds_cat - all_targets_cat
        signal_sum = np.sum(all_targets_cat)
        if signal_sum > 0:
            error = np.sum(np.abs(diff)) / signal_sum
            bias = np.sum(diff) / signal_sum
            self.log("val_error", error, prog_bar=True)
            self.log("val_bias", bias, prog_bar=True)
        else:
            error, bias = 0.0, 0.0

        score = corr_float - 0.01 * abs(bias)  # composite criterion: corr - 0.01*|bias|
        self.log("val_score", score, prog_bar=True, sync_dist=True)

        if score > self.best_val_metric:
            self.best_val_metric = score
            self.best_val_corr = corr_float
            self.best_val_epoch = epoch + 1

        # Compare against the single persistent global-best file across runs.
        pred_stats = {"min": float(all_preds_raw.min()), "max": float(all_preds_raw.max()), "mean": float(all_preds_raw.mean())}
        target_stats = {"min": float(all_targets_cat.min()), "max": float(all_targets_cat.max()), "mean": float(all_targets_cat.mean())}
        self._maybe_update_global_best_record(corr_float, error, bias, pred_stats=pred_stats, target_stats=target_stats)

        tag = getattr(self, '_run_tag', '')
        tag_str = f"  [{tag}]" if tag else ""
        print(f"\n[Eval Epoch {epoch + 1}]{tag_str} corr={corr_float:.4f}  error={error:.4f}  bias={bias:.4f}  score={score:.4f}  |  best: score={self.best_val_metric:.4f} @ epoch {self.best_val_epoch}")
        print(f"  pred -> min={all_preds_raw.min():.4f}  max={all_preds_raw.max():.4f}  mean={all_preds_raw.mean():.4f}")
        print(f"  target -> min={all_targets_cat.min():.4f}  max={all_targets_cat.max():.4f}  mean={all_targets_cat.mean():.4f}\n")

        if os.environ.get("SPIKESSM_DISABLE_EVAL_PLOT", "0") != "1":
            self._save_eval_plot(
                all_trial_inputs,
                all_trial_preds,
                all_trial_targets,
                all_preds_cat,
                all_targets_cat,
                epoch,
                trial_stds=all_trial_std,
            )

        self.val_data.clear()

    def _save_eval_plot(self, trial_inputs, trial_preds, trial_targets, all_preds, all_targets, epoch, trial_stds=None):
        try:
            base_dir = hydra.utils.get_original_cwd()
        except Exception:
            base_dir = os.getcwd()

        save_dir = os.path.join(base_dir, "result", self._name, "eval_plots")
        os.makedirs(save_dir, exist_ok=True)

        
        from scipy.stats import pearsonr
        trial_corrs = []
        for i, (pred_i, gt_i) in enumerate(zip(trial_preds, trial_targets)):
            p = pred_i[:3000]
            g = gt_i[:3000]
            if len(p) > 2 and np.std(p) > 1e-8 and np.std(g) > 1e-8:
                r3000, _ = pearsonr(p, g)
            else:
                r3000 = float("nan")

            if len(pred_i) > 2 and np.std(pred_i) > 1e-8 and np.std(gt_i) > 1e-8:
                r_full, _ = pearsonr(pred_i, gt_i)
            else:
                r_full = float("nan")

            trial_corrs.append((i, r3000, r_full))

        left_ids = list(range(min(3, len(trial_preds))))
        valid_corrs = [(i, r3000) for i, r3000, _ in trial_corrs if np.isfinite(r3000)]
        if len(valid_corrs) == 0:
            right_ids = list(range(min(3, len(trial_preds)), min(6, len(trial_preds))))
        else:
            sorted_by_corr = sorted(valid_corrs, key=lambda x: x[1])
            used = set(left_ids)
            right_ids = [i for i, _ in sorted_by_corr if i not in used][:3]
            if len(right_ids) < 3:
                for i in range(len(trial_preds)):
                    if i not in used and i not in right_ids:
                        right_ids.append(i)
                    if len(right_ids) == 3:
                        break

        slot_ids = left_ids + right_ids
        n_show = min(len(slot_ids), 6)
        fig, axes = plt.subplots(
            7, 2,
            figsize=(18, 18),
            gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1, 1.4]},
        )

        for i in range(6):
            col = 0 if i < 3 else 1
            row_base = (i % 3) * 2
            ax_ca = axes[row_base, col]
            ax_sp = axes[row_base + 1, col]

            if i >= n_show:
                ax_ca.axis("off")
                ax_sp.axis("off")
                continue

            tid = slot_ids[i]
            pred = np.clip(trial_preds[tid][:3000], 0, None)
            gt = trial_targets[tid][:3000]
            t = np.arange(len(pred))

            
            if tid < len(trial_inputs):
                ca = trial_inputs[tid][:3000]
                ax_ca.plot(np.arange(len(ca)), ca, color="olivedrab", linewidth=0.7)
            ax_ca.set_ylabel("ΔF (z-score)", fontsize=8)
            side = "left" if i < 3 else "worst"
            if i < 3:
                ax_ca.set_title(f"Trial {tid+1} — calcium input", fontsize=9)
            else:
                ax_ca.set_title(f"Trial {tid+1} ({side}) — calcium input", fontsize=9)
            ax_ca.set_xticklabels([])

            
            ax_sp.plot(t, gt, color="steelblue", linewidth=0.8, label="ground truth")
            ax_sp.plot(t, pred, color="tomato", linewidth=0.8, alpha=0.85, label="predicted")
            if trial_stds is not None and tid < len(trial_stds):
                std_i = trial_stds[tid][:3000]
                band_lo = np.clip(pred - 2 * std_i, 0, None)
                band_hi = pred + 2 * std_i
                ax_sp.fill_between(t, band_lo, band_hi,
                                   color="tomato", alpha=0.15, label="±2σ")
            if len(pred) > 2 and np.std(pred) > 1e-8 and np.std(gt) > 1e-8:
                r3000, _ = pearsonr(pred, gt)
            else:
                r3000 = float("nan")

            pred_full = trial_preds[tid]
            gt_full = trial_targets[tid]
            if len(pred_full) > 2 and np.std(pred_full) > 1e-8 and np.std(gt_full) > 1e-8:
                r_full, _ = pearsonr(pred_full, gt_full)
            else:
                r_full = float("nan")
            if i < 3:
                ax_sp.set_title(f"Trial {tid+1}  corr3000={r3000:.3f}  corr_full={r_full:.3f}", fontsize=9)
            else:
                ax_sp.set_title(f"Trial {tid+1} ({side})  corr3000={r3000:.3f}  corr_full={r_full:.3f}", fontsize=9)
            ax_sp.legend(fontsize=7, loc="upper right")
            ax_sp.set_ylabel("spike rate", fontsize=8)

        
        ax_input = axes[-1, 0]
        ax_gt = axes[-1, 1]
        finite_full = [(i, rf) for i, _, rf in trial_corrs if np.isfinite(rf)]
        if len(finite_full) > 0:
            worst_tid, worst_r_full = min(finite_full, key=lambda x: x[1])
            pred_full = np.clip(trial_preds[worst_tid], 0, None)
            gt_full = trial_targets[worst_tid]
            t_full = np.arange(len(pred_full))

            # Left: input calcium trace
            if worst_tid < len(trial_inputs):
                ca_full = trial_inputs[worst_tid]
                ax_input.plot(np.arange(len(ca_full)), ca_full, color="olivedrab", linewidth=0.6)
            ax_input.set_ylabel("ΔF", fontsize=8)

            # Right: GT (blue) + Pred (red) overlaid
            ax_gt.plot(t_full, gt_full, color="steelblue", linewidth=0.8, label="GT")
            ax_gt.plot(t_full, pred_full, color="tomato", linewidth=0.8, alpha=0.8, label="Pred")
            if trial_stds is not None and worst_tid < len(trial_stds):
                std_worst = trial_stds[worst_tid]
                band_lo = np.clip(pred_full - 2 * std_worst, 0, None)
                band_hi = pred_full + 2 * std_worst
                ax_gt.fill_between(t_full, band_lo, band_hi,
                                   color="tomato", alpha=0.12, label="±2σ")
            ax_gt.legend(fontsize=7, loc="upper right")

            p3000 = pred_full[:3000]
            g3000 = gt_full[:3000]
            if len(p3000) > 2 and np.std(p3000) > 1e-8 and np.std(g3000) > 1e-8:
                worst_r3000, _ = pearsonr(p3000, g3000)
            else:
                worst_r3000 = float("nan")

            ax_input.set_title(
                f"Worst trial #{worst_tid+1} input (len={len(pred_full)})  corr_full={worst_r_full:.3f}  corr3000={worst_r3000:.3f}",
                fontsize=9,
            )
            ax_gt.set_title(f"Worst trial #{worst_tid+1} GT vs Pred", fontsize=9)
            ax_input.set_xlabel("time", fontsize=8)
            ax_gt.set_xlabel("time", fontsize=8)
            ax_gt.set_ylabel("spike rate", fontsize=8)
        else:
            ax_input.axis("off")
            ax_gt.axis("off")
            ax_input.text(0.02, 0.95, "No finite full-trial corr available", va="top", ha="left", fontsize=9)

        var_tag = "_variance" if trial_stds is not None else ""
        fig.suptitle(f"Epoch {epoch + 1}{' (with variance)' if trial_stds is not None else ''}", fontsize=11)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"eval_epoch_{epoch + 1:03d}{var_tag}.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"  → eval plot saved: {save_path}")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss, preds, targets, weight, meta = self.model_step(batch)
        self.log("loss of test", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


        self.test_data.append({
            "preds": preds.detach().cpu().numpy(),
            "targets": targets.detach().cpu().numpy(),
            #"weights": weight.detach().cpu().numpy() if weight is not None else None,
            "meta": meta
        })


    def on_test_epoch_end(self) -> None:
        ""

        if len(self.test_data) == 0:
            return

        all_trial_preds, all_trial_targets, predict_trials_dict = self._trial_reconstruction(data=self.test_data)

        all_preds_cat = np.clip(np.concatenate(all_trial_preds), 0, None)
        all_targets_cat = np.concatenate(all_trial_targets)
        all_preds_raw = np.concatenate(all_trial_preds)

        corr_float = self._pearson_corr_np(all_preds_raw, all_targets_cat)
        self.log("corr", corr_float, prog_bar=True)

        diff = all_preds_cat - all_targets_cat
        signal_sum = np.sum(all_targets_cat)
        
        if signal_sum > 0:
            error = np.sum(np.abs(diff)) / signal_sum
            bias = np.sum(diff) / signal_sum
            self.log("error", error, prog_bar=True)
            self.log("bias", bias, prog_bar=True)
        else:
            error, bias = 0.0, 0.0

        try:
            base_dir = hydra.utils.get_original_cwd()
        except Exception:
            base_dir = os.getcwd()

        save_dir = os.path.join(base_dir, "result", self._name, "benchmark_results")
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"{self._name}_predictions.npz")
        np.savez(
            save_path,
            predictions=all_preds_cat,
            ground_truth=all_targets_cat,
            trials_preds=np.array(all_trial_preds, dtype=object),
            trials_targets=np.array(all_trial_targets, dtype=object),
            trial_ids=np.array(list(predict_trials_dict.keys())),
            correlation=float(corr_float),
            error=float(error),
            bias=float(bias),
            model_name=self._name
        )

        self.test_data.clear()


    def _trial_reconstruction(self, data=None, use_input=False, field=None, pre_transform=None) -> tuple:
        ""
        if data is None:
            data = self.test_data

        predict_trials = {}

        for batch_data in data:
            if field is not None:
                batch_preds = batch_data[field]
            elif use_input:
                batch_preds = batch_data["inputs"]
            else:
                batch_preds = batch_data["preds"]
            batch_gt = batch_data["targets"]
            meta = batch_data["meta"]
            # batch_weights = batch_data["weights"]
            
            if meta is None:
                continue
                
            for i in range(len(batch_preds)):
                tid = meta["trial_id"][i]
                start = meta["start_idx"][i].item()
                orig_len = meta["trial_len"][i].item()
                sample_rate = meta["sampling_rate"][i].item()

                pred_window = batch_preds[i]
                # input shape: (seq_len, feat) — take first channel; pred/target: (seq_len, 1)
                if pred_window.ndim >= 2:
                    pred_window = pred_window[:, 0]
                else:
                    pred_window = pred_window.squeeze()
                target_window = batch_gt[i].squeeze()
                
                end = start + pred_window.shape[0]
                # weight_window = batch_weights[i].squeeze() if batch_weights is not None else np.ones_like(pred_window)

                valid_start = start + self.context_left
                valid_end = end - self.context_right
                new_end = -self.context_right if self.context_right > 0 else None
                valid_pred = pred_window[self.context_left : new_end]
                if pre_transform is not None:
                    valid_pred = pre_transform(valid_pred)
                valid_target = target_window[self.context_left : new_end]
                
                if tid not in predict_trials:
                    predict_trials[tid] = {
                        "pred_sum": np.zeros(orig_len, dtype=np.float32),
                        "target_sum": np.zeros(orig_len, dtype=np.float32), 
                        "count": np.zeros(orig_len, dtype=np.float32),
                        "sample_rate": np.zeros(orig_len, dtype=np.float32)
                    }
                
                predict_trials[tid]["pred_sum"][valid_start:valid_end] += valid_pred
                predict_trials[tid]["target_sum"][valid_start:valid_end] += valid_target
                predict_trials[tid]["sample_rate"][valid_start:valid_end] = sample_rate
                predict_trials[tid]["count"][valid_start:valid_end] += 1.0

                
                # predict_trials[tid]["pred_sum"][start:end] += (pred_window * weight_window)
                # predict_trials[tid]["target_sum"][start:end] += (target_window * weight_window)

                # smoothing_samples = int(self.smoothing * sample_rate)
                # smooth_pred = smooth_spikes(
                #    pred_window, smoothing=smoothing_samples)


        all_trial_preds = []
        all_trial_targets = []
        #all_trial_preds_smooth = []
        #all_trial_targets_smooth = []

        for tid, data in predict_trials.items():
            counts = data["count"]
            counts[counts == 0] = 1.0
            
            final_pred = data["pred_sum"] / counts   
            final_target = data["target_sum"] / counts
            
            all_trial_preds.append(final_pred)
            all_trial_targets.append(final_target)

            #smooth_pred = gaussian_filter(final_pred.astype(float), sigma=sigma_frames)
            #smooth_target = gaussian_filter(final_target.astype(float), sigma=sigma_frames)
            
            #all_trial_preds_smooth.append(smooth_pred)
            #all_trial_targets_smooth.append(smooth_target)

        return all_trial_preds, all_trial_targets, predict_trials


    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            lr_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
            
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_config["monitor"] = "val_corr"
                lr_config["strict"] = True
            else:
                lr_config["monitor"] = "loss of train"
            return {"optimizer": optimizer, "lr_scheduler": lr_config}
        return {"optimizer": optimizer}
