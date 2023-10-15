import os
import argparse
import time
from datetime import timedelta
import pandas as pd
from utils.utils import *
from data_provider import data_provider
from mask_model import Model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", type=str, default="NIPS_TS_GECCO", help="SMD,SMAP,NIPS_TS_GECCO"
)
parser.add_argument(
    "--root_path", type=str, default="/workspace/ptm_anomaly_detection/dataset", help=""
)
parser.add_argument(
    "--save_name", type=str, default="GECCO100P1MMaskR320Finetune", help=""
)
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="The gpu no. used for training and inference (defaults to 0)",
)
parser.add_argument("--batch_size", type=int, default=32, help="")
parser.add_argument(
    "--lr", type=float, default=0.0001, help="The learning rate (defaults to 0.0001)"
)
parser.add_argument("--win_size", type=int, default=100, help="")
parser.add_argument("--num_workers", type=int, default=8, help="")
parser.add_argument("--patch_len", type=int, default=1, help="")
parser.add_argument("--mask_mode", type=str, default="M_binomial", help="")
parser.add_argument(
    "--repr_dims",
    type=int,
    default=320,
    help="The representation dimension (defaults to 320)",
)
parser.add_argument("--ratio", type=int, default=1, help="")
parser.add_argument("--iters", type=int, default=None, help="The number of iterations")
parser.add_argument("--epochs", type=int, default=None, help="The number of epochs")
parser.add_argument(
    "--show_every_iters", type=int, default=50, help="show the loss every iterations"
)
parser.add_argument("--seed", type=int, default=1, help="The random seed")
parser.add_argument(
    "--model_path",
    type=str,
    default="/workspace/ptm_anomaly_detection/finetune1/GECCOW100P1MMaskR320-20231014/model.pkl",
    help="",
)
args = parser.parse_args()
print(args)

device = init_dl_program(args.gpu, seed=args.seed)
config = dict(
    patch_len=args.patch_len,
    output_dims=args.repr_dims,
    hidden_dims=64,
    depth=10,
    win_size=args.win_size,
    mask_mode=args.mask_mode,
    device=device,
    lr=args.lr,
    show_every_iters=args.show_every_iters,
    after_iter_callback=None,
    after_epoch_callback=None,
)
run_dir = "detection/" + name_with_datetime(args.save_name)

os.makedirs(run_dir, exist_ok=True)

model = Model(**config)
train_dataset, train_data_loader = data_provider(args, "train", finetune=True)
test_dataset, test_data_loader = data_provider(args, "test")
print("trian_data_loader len: ", len(train_data_loader))
print("test_data_loader len: ", len(test_data_loader))

t = time.time()
model.load(args.model_path)
train_scores, _ = model.cal_scores(train_data_loader)
test_scores, test_labels = model.cal_scores(test_data_loader)
threshold = model.get_threshold([train_scores, test_scores], ratio=args.ratio)
pred = model.test(
    test_scores=test_scores,
    test_labels=test_labels,
    threshold=threshold,
    save_path=run_dir,
    verbose=True,
)

t = time.time() - t
print(f"\nRunning time: {timedelta(seconds=t)}\n")

# vis
vis(train_scores, save_path=run_dir, save_name="train_scores", threshold=threshold)
vis(test_scores, save_path=run_dir, save_name="test_socres", threshold=threshold)
vis(test_labels, save_path=run_dir, save_name="test_labels", threshold=None)
vis(pred, save_path=run_dir, save_name="pred", threshold=None)
print("Done\n")
