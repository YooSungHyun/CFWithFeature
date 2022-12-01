import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta
from setproctitle import setproctitle
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from model import DropNet
from datamodule import DropNetDataModule


def main(hparams):
    wandb_logger = WandbLogger(project="test", name="NCF_DropNet", save_dir="./")
    setproctitle("")
    pl.seed_everything(hparams.seed)

    ncf_datamodule = DropNetDataModule(hparams)
    model = DropNet(hparams)
    model.load_state_dict(
        torch.load(
            "",
            map_location="cuda",
        ),
        strict=False,
    )
    wandb_logger.watch(model, log="all")
    hparams.logger = wandb_logger

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.output_dir,
        save_top_k=3,
        mode="max",
        monitor="score",
        filename="test-{epoch:02d}-{val_loss:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    hparams.callbacks = [checkpoint_callback, lr_monitor]

    hparams.strategy = DDPStrategy(timeout=timedelta(days=30))
    trainer = pl.Trainer.from_argparse_args(hparams)
    trainer.fit(model, datamodule=ncf_datamodule)
    checkpoint_callback.best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", default=None, type=int, help="all seed")
    parser.add_argument("--local_rank", type=int, help="ddp local rank")
    parser.add_argument("--data_dir", type=str, help="target pytorch lightning data dirs")
    parser.add_argument("--ratio", type=float, help="train/valid split ratio")
    parser.add_argument("--output_dir", type=str, help="model output path")
    parser.add_argument("--num_proc", type=int, default=None, help="how many proc map?")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument(
        "--warmup_ratio", default=0.2, type=float, help="learning rate scheduler warmup ratio per EPOCH"
    )
    parser.add_argument("--max_lr", default=0.01, type=float, help="lr_scheduler max learning rate")
    parser.add_argument("--final_div_factor", default=1e4, type=int, help="(max_lr/25)*final_div_factor is final lr")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="weigth decay")
    parser.add_argument(
        "--per_device_train_batch_size",
        default=1,
        type=int,
        help="The batch size per GPU/TPU core/CPU for training.",
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        default=1,
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--n_users",
        default=200,
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--n_items",
        default=30000,
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--dropout",
        default=0.05,
        type=float,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--emb_dim",
        default=256,
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--layer_dim",
        default=256,
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--n_items_features",
        default=300,
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--n_users_features",
        default=200,
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    args = parser.parse_args()
    main(args)
