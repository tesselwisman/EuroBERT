import fire
import torch.distributed as dist
import torch.distributed.elastic.multiprocessing

from optimus.trainer.configuration.configs import Config
from optimus.trainer.data import Data, patch_spanner
from optimus.trainer.distributed import Distributed
from optimus.trainer.model.load import load_model, load_tokenizer
from optimus.trainer.pretrain import Pretrain
import wandb
import os


WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "EuroBERT-training")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")

@torch.distributed.elastic.multiprocessing.errors.record
def main(**kwargs):
    # Load configurations
    config = Config(**kwargs)
    print(config.system)
    if config.train.wandb:
        if not WANDB_API_KEY:
            print("WARNING: No WANDB api key found. This run will not be logged.")
        if config.is_main_process:
            wandb.login(key=WANDB_API_KEY)
            wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
    distributed = None

    if config.use_ddp or config.use_fsdp:
        distributed = Distributed(config)

    # Load/set model and get tokenizer.
    model = load_model(config)
    tokenizer = load_tokenizer(config)
    if distributed:
        dist.barrier()
        config.log_print("Model and tokenizer loaded on all rank.")

    # Setup model for distributed training
    if config.use_fsdp:
        config.log_print("Shared model training with FSDP.")
        model = distributed.fsdp_setup_model(model)
    elif config.use_ddp:
        config.log_print("Distributed model training with DDP.")
        model = distributed.ddp_setup_model(model)

    patch_spanner()
    if distributed:
        dist.barrier()
    config.log_print("Mosaic ML Streaming spanner patched successfully.")

    # Load data
    data = Data(config, tokenizer)
    if distributed:
        dist.barrier()
    config.log_print("Data loaded.")

    # Train model
    pretrain = Pretrain(model, data, distributed, config)
    pretrain.train()

    # Cleanup distributed training
    if distributed:
        distributed.cleanup()
    config.log_print("Training completed successfully.")

    exit(0)


if __name__ == "__main__":
    print("Running main...")
    fire.Fire(main)

