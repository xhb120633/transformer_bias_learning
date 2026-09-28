"""A minimal decoder-only Transformer implemented with PyTorch primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 106
    max_sequence_length: int = 401
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class CausalTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.d_model
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        zero_token_embedding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        sequence_length = tokens.shape[1]
        if sequence_length > self.config.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")
        positions = torch.arange(sequence_length, device=tokens.device)
        token_hidden = self.token_embedding(tokens)
        if zero_token_embedding_mask is not None:
            if zero_token_embedding_mask.shape != tokens.shape:
                raise ValueError("zero_token_embedding_mask must match tokens")
            token_hidden = token_hidden.masked_fill(
                zero_token_embedding_mask.unsqueeze(-1), 0.0
            )
        hidden = token_hidden + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=tokens.device,
            ),
            diagonal=1,
        )
        hidden = self.blocks(hidden, mask=causal_mask)
        return self.output(self.final_norm(hidden))
