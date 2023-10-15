import torch
import torch.nn as nn
import numpy as np
from models.mask_model import MaskModel
from utils.utils import EarlyStopping, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


class Model:
    def __init__(
        self,
        patch_len=1,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        win_size=100,
        mask_mode="M_binomial",
        device="cuda",
        lr=0.001,
        show_every_iters=100,
        after_iter_callback=None,
        after_epoch_callback=None,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.win_size = win_size
        self.device = device
        self.lr = lr
        self.show_every_iters = show_every_iters
        self.mask_mode = mask_mode
        self._net = MaskModel(
            patch_len=patch_len,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            win_size=win_size,
            mask_mode=mask_mode,
        ).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

        self.n_epochs = 0
        self.n_iters = 0

    def fit(self, train_loader, n_epochs=None, n_iters=None, verbose=False):
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        repr_sim = nn.MSELoss()
        restruction_error = nn.L1Loss()
        loss_log = []
        loss_log_iters = []
        self._net.train()
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            cum_loss = 0
            n_epoch_iters = 0
            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x = batch[0]
                x = x.to(self.device)

                optimizer.zero_grad()

                restructed_x, repr1, repr2 = self._net(x)
                repr_loss = repr_sim(repr1, repr2)
                restructed_loss = restruction_error(x, restructed_x)
                loss = repr_loss + restructed_loss * 2
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

                cum_loss += loss.item()
                loss_log_iters.append(loss.item())
                n_epoch_iters += 1
                self.n_iters += 1

                if self.n_iters % self.show_every_iters == 0:
                    print(f"iter #{self.n_iters}: loss={loss.item()}")

                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

            if interrupted:
                break

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

        return loss_log, loss_log_iters

    def finetune(self, train_loader, val_loader, save_path, finetune, verbose=False):
        # loss = repr_loss + restructed_loss * 2
        early_stopping = EarlyStopping(verbose=verbose)
        if finetune:
            # freeze encoder
            if verbose: print('freeze encoder')
            for param in self._net.encoder.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(self._net.decoder.parameters(), lr=self.lr)
        else:
            if verbose: print('freeze none')
            optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)

        repr_sim = nn.MSELoss()
        restruction_error = nn.L1Loss()
        train_loss_log = []
        train_loss_log_iters = []
        val_loss_log = []
        val_loss_log_iters = []
        while True:
            if early_stopping.early_stop:
                break
            cum_loss = 0
            n_epoch_iters = 0
            self._net.train()
            for batch in train_loader:
                x = batch[0]
                x = x.to(self.device)
                optimizer.zero_grad()
                restructed_x, repr1, repr2 = self._net(x)
                repr_loss = repr_sim(repr1, repr2)
                restructed_loss = restruction_error(x, restructed_x)
                loss = repr_loss + restructed_loss * 2
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                cum_loss += loss.item()
                train_loss_log_iters.append(loss.item())
                n_epoch_iters += 1
                self.n_iters += 1
                if self.n_iters % self.show_every_iters == 0:
                    print(f"train iter #{self.n_iters}: loss={loss.item()}")
            cum_loss /= n_epoch_iters
            train_loss_log.append(cum_loss)

            # val
            self.net.eval()
            val_loss = 0
            val_iters = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0]
                    x = x.to(self.device)
                    restructed_x, repr1, repr2 = self.net(x, mask=self.mask_mode)
                    repr_loss = repr_sim(repr1, repr2)
                    restructed_loss = restruction_error(x, restructed_x)
                    loss = repr_loss + restructed_loss * 2
                    val_loss += loss.item()
                    val_loss_log_iters.append(loss.item())
                    val_iters += 1
                val_loss /= val_iters
                val_loss_log.append(val_loss)

            if verbose:
                print(
                    f"Epoch #{self.n_epochs}: train_loss={cum_loss}, val_loss={val_loss}"
                )
            self.n_epochs += 1
            early_stopping(val_loss, self.net, save_path)

        return train_loss_log, train_loss_log_iters, val_loss_log, val_loss_log_iters

    def test(self, test_scores, test_labels, threshold, save_path, verbose=False):
        if verbose:
            print("Threshold :", threshold)
        pred = (test_scores > threshold).astype(int)
        gt = test_labels.astype(int)
        # detection adjustment
        gt, pred = adjustment(gt, pred)
        pred = np.array(pred)
        gt = np.array(gt)
        if verbose:
            print("pred: ", pred.shape)
            print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average="binary"
        )
        if verbose:
            print(
                "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                    accuracy, precision, recall, f_score
                )
            )

        f = open(f"{save_path}/result_anomaly_detection.txt", "a")
        f.write(
            f"patch_len:{self.patch_len}=output_dims:{self.output_dims}=hidden_dims:{self.hidden_dims}=depth:{self.depth}=win_size:{self.win_size}=mask_mode:{self.mask_mode}=threshold:{threshold}"
            + "\n"
        )
        f.write(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score
            )
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return pred

    def get_threshold(self, scores_list, ratio=1):
        scores = np.concatenate(scores_list, axis=0)
        threshold = np.percentile(scores, 100 - ratio)
        return threshold

    def cal_scores(self, loader):
        '''cal anomaly scores, scores = restructed_err*a

        Args:
            loader (_type_): data_loader

        Returns:
            (np.array, np.array): (scores, labels)
        '''
        scores = []
        labels = []
        repr_sim = nn.MSELoss(reduction="none")
        restruction_error = nn.L1Loss(reduction="none")

        self.net.eval()
        for i, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            restructed_x, repr1, repr2 = self.net(batch_x, mask=self.mask_mode)
            # criterion
            repr_err = (
                repr_sim(repr1, repr2)
                .mean(dim=-1)
                .mean(dim=-1)
                .mean(dim=-1)
                .unsqueeze(dim=-1)
            )  # b x input_dims x patch_num x co => b x 1
            restructed_err = restruction_error(batch_x, restructed_x).mean(
                dim=-1
            )  # b x t
            # score = restructed_err 
            score = repr_err+restructed_err*2 #
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(batch_y)

        scores = np.concatenate(scores, axis=0).reshape(-1)
        scores = np.array(scores)

        labels = np.concatenate(labels, axis=0).reshape(-1)
        labels = np.array(labels)

        return scores, labels

    def save(self, fn):
        """Save the model to a file.

        Args:
            fn (str): filename.
        """
        torch.save(self.net.state_dict(), fn)

    def load(self, fn):
        """Load the model from a file.

        Args:
            fn (str): filename.
        """
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
