import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

class BiMambaEncoder(nn.Module):
    def __init__(self, d_model, n_state):
        super(BiMambaEncoder, self).__init__()
        self.d_model = d_model
        
        self.mamba = Mamba(d_model, n_state)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # Residual connection of the original input
        residual = x
        
        # Forward Mamba
        x_norm = self.norm1(x)
        mamba_out_forward = self.mamba(x_norm)

        # Backward Mamba
        x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        mamba_out_backward = self.mamba(x_flip)
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # Combining forward and backward
        mamba_out = mamba_out_forward + mamba_out_backward
        
        mamba_out = self.norm2(mamba_out)
        ff_out = self.feed_forward(mamba_out)

        output = ff_out + residual
        return output

# # Initialize and test the model
# d_model = 512
# n_state = 64
# model = BiMambaEncoder(d_model, n_state).cuda()
# x = torch.rand(32, 100, d_model).cuda()  # Analog input data: (batch_size, seq_len, feature_dim)
# output = model(x)
# print(output.shape)  # Mamba Out: (32, 100, 512)

