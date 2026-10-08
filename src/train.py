import pyrootutils


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "requirements.txt"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Optional, Tuple

import os
import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger

from src.utils import pylogger, utils
from src.utils import rich_utils
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

# Enable TF32 Tensor Cores on Ampere+ GPUs for ~20% free speedup
torch.set_float32_matmul_precision("medium")


def _save_loo_checkpoint(cfg, trainer):
    """Copy best checkpoint to ckpt/loo_pretrain/ with a descriptive name."""
    import shutil
    best_path = trainer.checkpoint_callback.best_model_path
    if not best_path:
        log.warning("No best checkpoint found, skipping LOO ckpt save")
        return

    base_dir = hydra.utils.get_original_cwd()
    loo_dir = os.path.join(base_dir, "ckpt", "loo_pretrain")
    os.makedirs(loo_dir, exist_ok=True)

    # Name: {model_name}_{data_config}.ckpt
    model_name = cfg.model.get("name", "model")
    choices = {}
    try:
        from hydra.core.hydra_config import HydraConfig
        choices = HydraConfig.get().runtime.choices
    except Exception:
        pass
    data_name = choices.get("data", "default")
    dest = os.path.join(loo_dir, f"{model_name}_{data_name}.ckpt")
    shutil.copy2(best_path, dest)
    print(f"\n[LOO] Saved pretrained checkpoint: {dest}\n")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> Optional[float]:

    if cfg.get("extras", {}).get("enforce_tags"):
        log.info("Enforcing tags! (Set 'extras.enforce_tags=False' to disable)")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    if cfg.get("extras", {}).get("print_config"):
        log.info("Config tree! (Set 'extras.print_config=False' to disable)")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)

    # seed
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    choices = {}
    try:
        from hydra.core.hydra_config import HydraConfig
        choices = HydraConfig.get().runtime.choices
    except Exception:
        pass
    data_name = choices.get("data", "default")
    freeze_mode = cfg.model.get("freeze_mode", "none")
    fewshot_k = cfg.data.get("fewshot_k", 0)
    run_tag = f"data={data_name}"
    if fewshot_k > 0:
        run_tag += f"  k={fewshot_k}  freeze={freeze_mode}"
    # Attach to model for use in eval prints
    model._run_tag = run_tag

    print(f"\n{'='*60}")
    print(f"  Model: {cfg.model.get('name', model.__class__.__name__)}")
    print(f"  Run:   {run_tag}")
    print(f"  Total params:     {total_params:>12,}")
    print(f"  Trainable params: {train_params:>12,}")
    print(f"  Size: {total_params * 4 / 1024**2:.2f} MB (fp32)")
    print(f"{'='*60}\n")

    # Callbacks
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Loggers
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Suppress verbose third-party loggers before Trainer init
    import logging as _logging
    for _name in ["stable_pretraining", "datasets"]:
        _logging.getLogger(_name).setLevel(_logging.WARNING)

    # Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Remove auto-registered callbacks from stable_pretraining entry point
    trainer.callbacks = [
        cb for cb in trainer.callbacks
        if not type(cb).__module__.startswith("stable_pretraining")
    ]

    # Logging Hyperpara
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)


    # train
    if cfg.get("train"):
        # Run validation before training to capture epoch-0 baseline
        # (important for few-shot: shows pretrained model's zero-shot performance)
        if cfg.get("model", {}).get("pretrained_ckpt", ""):
            log.info("Running epoch-0 validation (pretrained zero-shot baseline)...")
            trainer.validate(model=model, datamodule=datamodule)

        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

        # Save LOO pretrained checkpoint to dedicated folder for few-shot use
        if cfg.get("save_loo_ckpt", False):
            _save_loo_checkpoint(cfg, trainer)

    # test
    train_metrics = trainer.callback_metrics
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None

        
        print(f"============== DATAMODULE HPARAMS ==============")
        from pprint import pprint
        pprint(datamodule.hparams)
        print(f"================================================")
    
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    
    return train_metrics.get("loss of train")

if __name__ == "__main__":
    main()