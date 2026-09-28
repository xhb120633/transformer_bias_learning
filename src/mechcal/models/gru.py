"""Unidirectional session GRU; hidden state never crosses participants."""
from dataclasses import dataclass, asdict
import torch
from torch import nn

@dataclass
class GRUConfig:
    vocab_size: int = 106
    embedding_dim: int = 64
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)

class CausalGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.rnn = nn.GRU(config.embedding_dim, config.hidden_size,
                          config.num_layers, batch_first=True,
                          dropout=config.dropout if config.num_layers > 1 else 0.)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, tokens):
        states, _ = self.rnn(self.embedding(tokens))
        return self.head(states)
