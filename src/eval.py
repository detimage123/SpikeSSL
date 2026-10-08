import pyrootutils

# root
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "requirements.txt"],
    pythonpath=True,
    dotenv=True,
)

from typing import List
import hydra
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils import pylogger, utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    # ckpt file
    if not cfg.get("ckpt_path"):
        raise ValueError(
            "ckpt_path is not provided. "
            "Please specify it via command line, e.g., python src/eval.py ckpt_path=logs/.../*.ckpt"
        )

    # DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Loggers
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # callbacks
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    print(f"============== DATAMODULE HPARAMS ==============")
    from pprint import pprint
    pprint(datamodule.hparams)
    print(f"================================================")
    
    # test
    log.info("Starting evaluation...")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()