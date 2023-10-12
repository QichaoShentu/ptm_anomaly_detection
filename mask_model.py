import torch
import torch.nn as nn
from models.mask_model import MaskModel


class Model:
    def __init__(
        self,
        patch_len=1,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        win_size=10,
        mask_mode="M_binomial",
        device="cuda",
        lr=0.001,
        show_every_iters=100,
        after_iter_callback=None,
        after_epoch_callback=None,
    ):
        super().__init__()
        self.device = device
        self.lr = lr
        self.show_every_iters = show_every_iters
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
