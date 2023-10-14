from torch import nn
from layers.decoder import Decoder
from layers.encoder_cid import TSEncoder_CInd

class MaskModel(nn.Module):
    def __init__(
        self,
        patch_len=1,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        win_size=100,
        mask_mode="M_binomial",
    ):
        super().__init__()
        self.patch_len = patch_len
        patch_num = int((win_size - patch_len) / patch_len + 1)
        self.mask_mode=mask_mode
        self.encoder = TSEncoder_CInd(
            input_dims=patch_len,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            mask_mode=mask_mode,
        )
        self.decoder = Decoder(patch_num * output_dims, win_size)

    def forward(self, x, mask=None):  # b x T x input_dims
        x = x.permute(0, 2, 1)  # b x input_dims x T
        x = x.unfold(
            dimension=-1, size=self.patch_len, step=self.patch_len
        )  # b x input_dims x patch_num x patch_len
        repr1 = self.encoder(x, mask='all_true')  # b x input_dims x patch_num x co
        repr2 = self.encoder(x, mask=mask)
        x = self.decoder(repr1)  # b x win_size x input_dims
        return x, repr1, repr2
