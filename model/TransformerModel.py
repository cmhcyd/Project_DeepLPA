import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence of features."""
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    """
    Transformer Encoder model for tabular data.
    Each feature becomes a token in the sequence.
    Args:
        in_dim: number of input features (sequence length)
        num_classes: output dimension (for classification or regression)
        task: 'classification' or 'regression' (affects final activation, but not used here, just logits)
        d_model: embedding dimension for transformer (default: 64)
        nhead: number of attention heads (default: 4)
        num_layers: number of transformer encoder layers (default: 2)
        dim_feedforward: hidden dimension in FFN (default: 128)
        dropout: dropout rate (default: 0.1)
        activation: activation function 'relu' or 'gelu' (default: 'gelu')
        pos_encoding: whether to add positional encoding (default: True)
        use_cls_token: if True, add a learnable [CLS] token (like BERT); else use mean pooling (default: True)
    """
    def __init__(self,
                 in_dim: int,
                 num_classes: int,
                 task: str = "classification",
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 pos_encoding: bool = True,
                 use_cls_token: bool = True,
                 **kwargs):
        super().__init__()
        self.task = task
        self.in_dim = in_dim
        self.use_cls_token = use_cls_token
        self.d_model = d_model
        self.pos_encoding = pos_encoding

        self.input_proj = nn.Linear(1, d_model)
        if pos_encoding:
            self.pos_encoder = PositionalEncoding(d_model, max_len=in_dim + (1 if use_cls_token else 0))
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(-1)
        x = self.input_proj(x)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_encoding:
            x = self.pos_encoder(x)
        x = self.transformer(x)
        if self.use_cls_token:
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)

        logits = self.head(x)
        return logits