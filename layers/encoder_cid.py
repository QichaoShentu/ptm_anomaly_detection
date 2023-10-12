import torch
from torch import nn
import numpy as np
from .dilated_conv import DilatedConvEncoder


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t : t + l] = False
    return res


def generate_binomial_mask(M, B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(M, B, T))).to(torch.bool)


class TSEncoder_CInd(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_dims=64,
        depth=10,
        mask_mode="M_binomial",
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):  # b x input_dims x patch_num x patch_len 
        dims = x.shape[1] 
        x = x.reshape(-1, x.shape[-2], x.shape[-1]) # b*input_dims x patch_num x patch_len   
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        
        x = self.input_fc(x)  # b*input_dims x patch_num x h

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = "all_true"
        
        if mask == "M_binomial":
            # Mutil copied ts, mean repr
            M = 3
            mask = generate_binomial_mask(M, x.size(0), x.size(1)).to(x.device)
            mask &= nan_mask

            x_M = x.unsqueeze(dim=0).repeat(
                M, 1, 1, 1
            )  # M x b*input_dims x patch_num x hW
            x_M[~mask] = 0
            x_M = x_M.permute(0, 1, 3, 2)  # M x b*input_dims x h x patch_num
            x = [
                self.repr_dropout(self.feature_extractor(x_M[i])).unsqueeze(dim=0)
                for i in range(M)
            ]
            x = torch.cat(x, dim=0).mean(dim=0)  # b*input_dims x co x patch_num
            x = x.transpose(1, 2)  # b*input_dims x patch_num x co
            x = x.reshape(
                -1, dims, x.shape[-2], x.shape[-1]
            )  # b x input_dims x patch_num x co
            return x
        elif mask == 'binomial':
            mask = mask = generate_binomial_mask(1, x.size(0), x.size(1)).to(x.device)
            mask = mask.squeeze(dim=0)
        elif mask == "continuous":
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "all_true":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "all_false":
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == "mask_last":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # b*input_dims x h x patch_num
        x = self.repr_dropout(
            self.feature_extractor(x)
        )  # b*input_dims x co x patch_num
        x = x.transpose(1, 2)  # B*input_dims x patch_num x co
        x = x.reshape(
            -1, dims, x.shape[-2], x.shape[-1]
        )  # b x input_dims x patch_num x co
        return x
