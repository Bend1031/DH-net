import os
import shutil

import hydra
import numpy as np
import torch
import torch.optim as optim
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.config import Config
from lib.dataset import OSDataset
from lib.exceptions import NoGradientError
from lib.loss import loss_l2
from lib.model_L2 import L2Net


# 加载配置
@hydra.main(version_base=None, config_path="conf", config_name="config_asl")
def main(cfg: Config):
    # %%init cuda and seed
    # print(OmegaConf.to_yaml(cfg))

    # %% Tensorboard
    writer_train_loss = SummaryWriter(
        "./runs/" + str(cfg.dataset.checkpoint_prefix) + "/train_loss"
    )
    writer_valid_loss = SummaryWriter(
        "./runs/" + str(cfg.dataset.checkpoint_prefix) + "/valid_loss"
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Seed
    torch.manual_seed(1)
    if use_cuda:
        torch.cuda.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.benchmark = True  # type: ignore

    # cfg = parse_cfg()
    # print(cfg)

    # %% Create the folders
    # Create the folders for plotting if need be
    if cfg.plot:
        plot_path = "train_vis"
        os.makedirs(plot_path, exist_ok=True)
    os.makedirs(cfg.dataset.checkpoint_directory, exist_ok=True)
    print("Checkpoint directory created")

    if os.path.exists(cfg.log.log_file):
        print("Log is opening.")
    log_file = open(cfg.log.log_file, "a+")

    # %% load valid and train Dataset
    validation_dataloader = None
    if cfg.train.use_validation:
        validation_dataset = OSDataset(
            scene_list_path="datasets_utils/osdataset/test.txt",
            base_path=cfg.dataset.dataset_path,
            train=False,
            preprocessing=cfg.preprocessing,
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
        )
        validation_dataset.build_dataset()

    training_dataset = OSDataset(
        scene_list_path="datasets_utils/osdataset/train.txt",
        preprocessing=cfg.preprocessing,
        train=True,
    )

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    # %% Creating CNN model and optimizer
    # model = D2Net(model_file=cfg.model_file, use_cuda=use_cuda)
    model = L2Net().to(device)
    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr
    )

    # %% Resume training if needed
    # checkpoint 不存在，从头开始训练
    checkpoints = os.listdir(cfg.dataset.checkpoint_directory)
    if not checkpoints:
        start_epoch = 1
        train_loss_history = []
        validation_loss_history = []
    # checkpoint 存在，从checkpoint开始训练
    else:
        # exclude the best checkpoint
        checkpoints = [
            x
            for x in checkpoints
            if not x.startswith(cfg.dataset.checkpoint_prefix + ".best")
        ]

        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split(".")[-2]))  # type: ignore
        latest_checkpoint_path = os.path.join(
            cfg.dataset.checkpoint_directory, latest_checkpoint
        )
        checkpoint = torch.load(latest_checkpoint_path)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        print("=> loaded checkpoint (epoch {})".format(checkpoint["epoch_idx"]))

        start_epoch = checkpoint["epoch_idx"] + 1
        # Initialize the history
        train_loss_history = checkpoint["train_loss_history"]
        validation_loss_history = checkpoint["validation_loss_history"]
        # min_validation_loss = checkpoint["min_validation_loss"]

    # Define epoch function
    def process_epoch(
        epoch, model, loss_fn, optimizer, dataloader, device, log_file, is_train=True
    ):
        mode = "train" if is_train else "valid"
        epoch_losses = []

        torch.set_grad_enabled(is_train)
        print(f"{mode.capitalize()}ing epoch {epoch}")
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, batch in progress_bar:
            if is_train:
                # Forward pass
                optimizer.zero_grad()

            batch["train"] = is_train
            batch["epoch_idx"] = epoch
            batch["batch_idx"] = batch_idx
            batch["batch_size"] = cfg.train.batch_size
            batch["preprocessing"] = cfg.preprocessing
            batch["log_interval"] = cfg.log.log_interval

            try:
                loss = loss_fn(model, batch, device).to(device)
                loss.requires_grad_(True)
            except NoGradientError:
                continue

            current_loss = loss.data.cpu().numpy()
            epoch_losses.append(current_loss)

            progress_bar.set_postfix(loss=f"{np.mean(epoch_losses):.4f}")

            if batch_idx % cfg.log.log_interval == 0:
                log_file.write(
                    f"[{mode}] epoch {epoch} - batch {batch_idx} / {len(dataloader)} - avg_loss: {np.mean(epoch_losses):.4f}\n"
                )

            if is_train:
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 20)  # 在代码中加入这行实现梯度裁剪
                optimizer.step()

        writer_train_loss.add_scalar(
            "Loss", np.mean(epoch_losses), epoch
        ) if is_train else writer_valid_loss.add_scalar(
            "Loss", np.mean(epoch_losses), epoch
        )

        log_file.write(
            f"[{mode}] epoch {epoch} - avg_loss: {np.mean(epoch_losses):.4f}\n"
        )
        log_file.flush()

        return np.mean(epoch_losses)

    # Train loop

    train_loss_history = []
    validation_loss_history = []
    for epoch in range(start_epoch, start_epoch + cfg.train.num_epochs):
        training_dataset.build_dataset()
        train_loss_history.append(
            process_epoch(
                epoch,
                model,
                loss_l2,
                optimizer,
                training_dataloader,
                device,
                log_file,
            )
        )
        if cfg.train.use_validation:
            validation_loss_history.append(
                process_epoch(
                    epoch,
                    model,
                    loss_l2,
                    optimizer,
                    validation_dataloader,
                    device,
                    log_file,
                    is_train=False,
                )
            )

        # Save checkpoint
        checkpoint_path = os.path.join(
            cfg.dataset.checkpoint_directory,
            f"{cfg.dataset.checkpoint_prefix}.{epoch:02d}.pth",
        )
        checkpoint = {
            "args": cfg,
            "epoch_idx": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss_history": train_loss_history,
            "validation_loss_history": validation_loss_history,
            "lr": optimizer.param_groups[0]["lr"],
        }
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if cfg.train.use_validation and (len(validation_loss_history) == 1):
            min_validation_loss = validation_loss_history[0]

        if cfg.train.use_validation and (
            validation_loss_history[-1] == min(validation_loss_history)
        ):
            min_validation_loss = validation_loss_history[-1]
            best_checkpoint_path = os.path.join(
                cfg.dataset.checkpoint_directory,
                f"{cfg.dataset.checkpoint_prefix}.best.pth",
            )
            shutil.copy(checkpoint_path, best_checkpoint_path)
    log_file.close()
    print("Done")


if __name__ == "__main__":
    main()
