from torch import nn


class Decoder(nn.Module):
    def __init__(self, patch_num_co, win_size, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(patch_num_co, win_size)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # bx input_dims x patch_num x co
        x = self.flatten(x) # b x input_dims x patch_num*co
        x = self.linear(x) # b x input_dims x win_size
        x = self.dropout(x)
        x = x.permute(0, 2, 1) # b x win_size x input_dims
        return x
