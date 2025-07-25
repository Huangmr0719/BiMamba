import torch
import torch.nn as nn
from mamba_ssm import Mamba


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BiMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        share_ffn: bool = False,
        share_norm: bool = False,
    ):
        super().__init__()

        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )
        self.mamba_rev = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )

        if share_norm:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.ln1_rev = self.ln1
            self.ln2_rev = self.ln2
        else:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.ln1_rev = nn.LayerNorm(d_model)
            self.ln2_rev = nn.LayerNorm(d_model)

        # FFN
        if share_ffn:
            self.ffn = FeedForward(d_model, dropout=dropout)
            self.ffn_rev = self.ffn
        else:
            self.ffn = FeedForward(d_model, dropout=dropout)
            self.ffn_rev = FeedForward(d_model, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward_branch(self, x, mamba, ln1, ln2, ffn, flip_time=False):
        if flip_time:
            x_in = torch.flip(x, dims=[1])  # (B, S, D)
        else:
            x_in = x

        y = mamba(x_in)                           # Mamba
        y = self.dropout(y)

        if flip_time:
            y = torch.flip(y, dims=[1])
        y = ln1(x + y)                            # Add & Norm

        y2 = ffn(y)                                # FeedForward
        y2 = self.dropout(y2)
        y = ln2(y + y2)                            # Add & Norm

        return y

    def forward(self, x):
        """
        x: (B, S, D)
        return: (B, S, D)
        """
        out_fwd = self.forward_branch(
            x, self.mamba_fwd, self.ln1, self.ln2, self.ffn, flip_time=False
        )

        out_rev = self.forward_branch(
            x, self.mamba_rev, self.ln1_rev, self.ln2_rev, self.ffn_rev, flip_time=True
        )

        out = 0.5 * (out_fwd + out_rev) 
        return out


class BiMambaEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, **block_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [BiMambaBlock(d_model=d_model, **block_kwargs) for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_ln(x)


if __name__ == "__main__":
    B, S, D = 2, 128, 512
    x = torch.randn(B, S, D)

    model = BiMambaEncoder(
        d_model=D,
        num_layers=4,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        dropout=0.1,
        share_ffn=False,
        share_norm=False,
    )

    y = model(x)
    print(y.shape)  # torch.Size([2, 128, 512])
