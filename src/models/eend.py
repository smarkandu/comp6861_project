import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyEEND(nn.Module):
    """
    Lightweight EEND-style diarization model.

    Input:
        x: [B, T, F] log-mel features

    Output:
        logits: [B, T, S]
        where S = max number of speakers
    """

    def __init__(
        self,
        input_dim=80,
        hidden_dim=128,
        num_speakers=4,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_layer = nn.Linear(hidden_dim, num_speakers)

    def forward(self, x, padding_mask=None):
        """
        x: [B, T, F]
        padding_mask: [B, T], True for padded frames
        """
        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        logits = self.output_layer(h)
        return logits

@torch.no_grad()
def predict_eend(model, features, threshold=0.5, device="cuda"):
    model.eval()

    x = features.unsqueeze(0).to(device)  # [1, T, F]
    logits = model(x)

    probs = torch.sigmoid(logits).squeeze(0)
    activity = (probs >= threshold).int()

    return activity.cpu(), probs.cpu()