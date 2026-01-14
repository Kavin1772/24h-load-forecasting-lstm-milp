import torch
import torch.nn as nn

class Seq2HorizLSTM(nn.Module):
    """
    Input:  (B, T_in, F)
    Output: (B, T_out)
    """
    def __init__(self, in_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0, T_out: int = 24):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, T_out)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last = h_n[-1]
        return self.head(last)
