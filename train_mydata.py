import argparse
import os
import shutil
import warnings

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.dataset import MegaDepthDataset, QxslabSarOptDataset
from lib.exceptions import NoGradientError
from lib.loss import loss_function, loss_function_qxs
from lib.model import D2Net

#%%init cuda and seed
# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Seed
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
np.random.seed(1)

#%% Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument(
        "--dataset_path", type=str, required=True, help="path to the dataset"
    )
    # parser.add_argument(
    #     "--scene_info_path",
    #     type=str,
    #     required=True,
    #     help="path to the processed scenes",
    # )

    parser.add_argument(
        "--preprocessing",
        type=str,
        default="caffe",
        help="image preprocessing (caffe or torch)",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="models/d2_tf.pth",
        help="path to the full model",
    )

    parser.add_argument(
        "--num_epochs", type=int, default=10, help="number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers for data loading"
    )

    parser.add_argument(
        "--use_validation",
        dest="use_validation",
        action="store_true",
        help="use the validation split",
    )
    parser.set_defaults(use_validation=False)

    parser.add_argument(
        "--log_interval", type=int, default=250, help="loss logging interval"
    )

    parser.add_argument(
        "--log_file", type=str, default="log.txt", help="loss logging file"
    )

    parser.add_argument(
        "--plot", dest="plot", action="store_true", help="plot training pairs"
    )
    parser.set_defaults(plot=False)

    parser.add_argument(
        "--checkpoint_directory",
        type=str,
        default="checkpoints",
        help="directory for training checkpoints",
    )
    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default="d2",
        help="prefix for training checkpoints",
    )

    return parser.parse_args()


args = parse_args()


print(args)
#%% Create the folders
# Create the folders for plotting if need be
if args.plot:
    plot_path = "train_vis"
    if os.path.isdir(plot_path):
        print("[Warning] Plotting directory already exists.")
    else:
        os.mkdir(plot_path)
# Create the checkpoint directory
if os.path.isdir(args.checkpoint_directory):
    # print("Checkpoint directory already exists.")
    pass
else:
    os.mkdir(args.checkpoint_directory)
    print("Checkpoint directory created")

# Open the log file for writing
if os.path.exists(args.log_file):
    print("Log is opening.")
log_file = open(args.log_file, "a+")

#%% load valid and train Dataset
if args.use_validation:
    validation_dataset = QxslabSarOptDataset(
        scene_list_path="qxslab_utils/valid.txt",
        base_path=args.dataset_path,
        train=False,
        preprocessing=args.preprocessing,
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    validation_dataset.build_dataset()

training_dataset = QxslabSarOptDataset(
    scene_list_path="qxslab_utils/train.txt",
    preprocessing=args.preprocessing,
    train=True,
)

training_dataloader = DataLoader(
    training_dataset, batch_size=args.batch_size, num_workers=args.num_workers
)
#%% Creating CNN model and optimizer
model = D2Net(model_file=args.model_file, use_cuda=use_cuda)

# Optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
)

#%% Resume training if needed
# checkpoint 不存在，从头开始训练
checkpoints = os.listdir(args.checkpoint_directory)
if not checkpoints:
    start_epoch = 1
    train_loss_history = []
    validation_loss_history = []
# checkpoint 存在，从checkpoint开始训练
else:
    # exclude the best checkpoint
    checkpoints = [
        x for x in checkpoints if not x.startswith(args.checkpoint_prefix + ".best")
    ]

    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split(".")[-2]))  # type: ignore
    latest_checkpoint_path = os.path.join(args.checkpoint_directory, latest_checkpoint)
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
    epoch_idx,
    model,
    loss_function,
    optimizer,
    dataloader,
    device,
    log_file,
    args,
    train=True,
):
    epoch_losses = []

    torch.set_grad_enabled(train)
    if train:
        print("Training epoch %d" % epoch_idx)
    else:
        print("Validating epoch %d" % epoch_idx)
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, batch in progress_bar:
        if train:
            # Forward pass
            optimizer.zero_grad()

        batch["train"] = train
        batch["epoch_idx"] = epoch_idx
        batch["batch_idx"] = batch_idx
        batch["batch_size"] = args.batch_size
        batch["preprocessing"] = args.preprocessing
        batch["log_interval"] = args.log_interval

        try:
            loss = loss_function(model, batch, device, plot=args.plot)
        except NoGradientError:
            continue

        current_loss = loss.data.cpu().numpy()[0]
        epoch_losses.append(current_loss)

        progress_bar.set_postfix(loss=("%.4f" % np.mean(epoch_losses)))

        if batch_idx % args.log_interval == 0:
            log_file.write(
                "[%s] epoch %d - batch %d / %d - avg_loss: %f\n"
                % (
                    "train" if train else "valid",
                    epoch_idx,
                    batch_idx,
                    len(dataloader),
                    np.mean(epoch_losses),
                )
            )

        if train:
            loss.backward()
            optimizer.step()

    log_file.write(
        "[%s] epoch %d - avg_loss: %f\n"
        % ("train" if train else "valid", epoch_idx, np.mean(epoch_losses))
    )
    log_file.flush()

    return np.mean(epoch_losses)


#%%train
for epoch_idx in range(start_epoch, start_epoch + args.num_epochs):
    # Process epoch
    training_dataset.build_dataset()
    train_loss_history.append(
        process_epoch(
            epoch_idx,
            model,
            loss_function,
            optimizer,
            training_dataloader,
            device,
            log_file,
            args,
        )
    )

    if args.use_validation:

        validation_loss_history.append(
            process_epoch(
                epoch_idx,
                model,
                loss_function,
                optimizer,
                validation_dataloader,
                device,
                log_file,
                args,
                train=False,
            )
        )
        # Save the current checkpoint
    checkpoint_path = os.path.join(
        args.checkpoint_directory,
        "%s.%02d.pth" % (args.checkpoint_prefix, epoch_idx),
    )
    # 更新学习率,每3个epoch学习率减半
    if (epoch_idx) % 3 == 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.5
            print("learning rate is updated to {}".format(param_group["lr"]))

    checkpoint = {
        "args": args,
        "epoch_idx": epoch_idx,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss_history": train_loss_history,
        "validation_loss_history": validation_loss_history,
        "lr": optimizer.param_groups[0]["lr"]
        # "min_validation_loss": min_validation_loss,
    }
    torch.save(checkpoint, checkpoint_path)

    # Save the best checkpoint
    if args.use_validation and (len(validation_loss_history) == 1):
        min_validation_loss = validation_loss_history[0]

    if args.use_validation and (
        validation_loss_history[-1] == min(validation_loss_history)
    ):
        min_validation_loss = validation_loss_history[-1]
        best_checkpoint_path = os.path.join(
            args.checkpoint_directory, "%s.best.pth" % args.checkpoint_prefix
        )

        shutil.copy(checkpoint_path, best_checkpoint_path)

# Close the log file
log_file.close()
