import torch.nn as nn
from .config import ChemConfig

class EncoderBlock(nn.Module):
    def __init__(self, config: ChemConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.d_model, config.n_heads,
                                               dropout=config.dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.d_model)
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, padding_mask=None):
        # Pre-LayerNorm: Нормализация ПЕРЕД вниманием
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=padding_mask)
        x = residual + self.dropout(attn_out)

        # Pre-LayerNorm: Нормализация ПЕРЕД FFN
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, config: ChemConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.d_model, config.n_heads,
                                               dropout=config.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(config.d_model, config.n_heads,
                                                dropout=config.dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.d_model)
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, memory, tgt_mask=None, memory_padding_mask=None):
        # Pre-LayerNorm Self-Attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(query=x, key=x, value=x, attn_mask=tgt_mask)
        x = residual + self.dropout(attn_out)

        # Pre-LayerNorm Cross-Attention
        residual = x
        x = self.norm2(x)
        cross_out, _ = self.cross_attn(query=x, key=memory, value=memory,
                                       key_padding_mask=memory_padding_mask)
        x = residual + self.dropout(cross_out)

        # Pre-LayerNorm FFN
        residual = x
        x = self.norm3(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)
        return x
