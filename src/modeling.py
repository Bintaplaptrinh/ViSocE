"""Model definitions and factory for IEEE experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class PhoBERTMLP(nn.Module):
    def __init__(self, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        hidden_size = self.phobert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_token))


class PhoBERTMHAMLP(nn.Module):
    def __init__(self, num_labels: int, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        hidden_size = self.phobert.config.hidden_size
        self.mha = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mha_out, _ = self.mha(hidden, hidden, hidden)
        pooled = self.dropout(mha_out[:, 0, :])
        return self.classifier(pooled)


class PhoBERTDualStream(nn.Module):
    def __init__(self, num_labels: int, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        hidden_size = self.phobert.config.hidden_size

        self.mha = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        mha_out, _ = self.mha(hidden, hidden, hidden)
        mha_vec = mha_out[:, 0, :]

        cnn_in = hidden.transpose(1, 2)
        cnn_vec = self.cnn(cnn_in).squeeze(-1)

        fused = self.dropout(torch.cat([mha_vec, cnn_vec], dim=1))
        return self.classifier(fused)


class MultiVectorAttention(nn.Module):
    def __init__(self, num_labels: int, hidden_size: int, num_heads: int):
        super().__init__()
        self.label_queries = nn.Parameter(torch.randn(1, num_labels, hidden_size))
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, hidden_states):
        batch_size = hidden_states.size(0)
        q = self.label_queries.expand(batch_size, -1, -1)
        out, _ = self.attention(query=q, key=hidden_states, value=hidden_states)
        return out


class PhoBERTMVAKAN1D(nn.Module):
    def __init__(self, num_labels: int, num_heads: int = 8, dropout: float = 0.3, grid_size: int = 5):
        super().__init__()
        from kan_Arch_1D import KAN, KAN_Convolutional_Layer_1D

        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2", add_pooling_layer=False)
        hidden_size = self.phobert.config.hidden_size

        self.multi_vector = MultiVectorAttention(num_labels, hidden_size, num_heads)
        self.mva_proj = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
        )

        self.cnn_bottleneck = nn.Conv1d(in_channels=hidden_size, out_channels=16, kernel_size=1)
        self.kan_conv = nn.Sequential(
            KAN_Convolutional_Layer_1D(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=1,
                grid_size=grid_size,
            ),
            nn.AdaptiveMaxPool1d(1),
        )

        self.dropout = nn.Dropout(dropout)
        in_features = (num_labels * 64) + 16
        self.classifier = KAN(
            layers_hidden=[in_features, 64, num_labels],
            grid_size=grid_size,
            spline_order=3,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        mv_out = self.multi_vector(hidden)
        mv_proj = self.mva_proj(mv_out)
        mv_flat = mv_proj.reshape(mv_proj.size(0), -1)

        cnn_in = hidden.transpose(1, 2)
        cnn_bottle = self.cnn_bottleneck(cnn_in)
        kan_cnn_out = self.kan_conv(cnn_bottle).squeeze(-1)

        fused = self.dropout(torch.cat([mv_flat, kan_cnn_out], dim=1))
        return self.classifier(fused, update_grid=False)


def build_model(name: str, num_labels: int, dropout: float, num_heads: int, grid_size: int) -> nn.Module:
    registry = {
        "PhoBERT_MLP": lambda: PhoBERTMLP(num_labels=num_labels, dropout=dropout),
        "PhoBERT_MHA_MLP": lambda: PhoBERTMHAMLP(num_labels=num_labels, num_heads=num_heads, dropout=dropout),
        "PhoBERT_DualStream": lambda: PhoBERTDualStream(num_labels=num_labels, num_heads=num_heads, dropout=dropout),
        "PhoBERT_MVA_KAN_1D": lambda: PhoBERTMVAKAN1D(
            num_labels=num_labels,
            num_heads=num_heads,
            dropout=dropout,
            grid_size=grid_size,
        ),
    }

    if name not in registry:
        raise ValueError(f"Unsupported model: {name}")

    return registry[name]()
