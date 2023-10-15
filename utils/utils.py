import numpy as np
import pandas as pd
import torch
import random
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
plt.ioff()


def vis(scores, save_path, save_name, threshold=None):
    x = range(len(scores))
    y = scores
    linewidth = 0.1
    plt.plot(x, y, linewidth=linewidth)
    if threshold is not None:
        plt.axhline(threshold, color='r', linestyle='--', label=f'threshold={threshold:.4f}')
        plt.ylabel('score')
        plt.legend(loc="upper right")
    
    plt.savefig(f"{save_path}/{save_name}.png")
    plt.clf()
    


def save_log(log, flag, unit, run_dir):
    result = {
        f"{unit}": range(len(log)),
        "loss": log,
    }
    result = pd.DataFrame(result)
    result.to_csv(f"{run_dir}/{flag}_result_{unit}.csv", index=False)


def name_with_datetime(prefix="default"):
    now = datetime.now()
    return prefix + "-" + now.strftime("%Y%m%d")


def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None,
):
    import torch

    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)

    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)

    if isinstance(device_name, (str, int)):
        device_name = [device_name]

    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == "cuda":
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    return devices if len(devices) > 1 else devices[0]


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=3, verbose=False, delta=0):
        """

        Args:
            patience (int, optional): how many epochs to wait before stopping when loss is
               not improving. Defaults to 7.
            verbose (bool, optional): _description_. Defaults to False.
            delta (int, optional): minimum difference between new loss and old loss for
               new loss to be considered as an improvement. Defaults to 0.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + f"model.pkl")
        self.val_loss_min = val_loss


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred
