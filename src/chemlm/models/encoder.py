import torch
import torch.nn as nn
from .blocks import EncoderBlock

class ChemEncoder(nn.Module):
    def __init__(self, config, embed_layer, pos_embed_layer):
        super().__init__()
        self.embed = embed_layer
        self.pos_embed = pos_embed_layer
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, src_tokens, padding_mask=None):
        B, T = src_tokens.size()
        positions = torch.arange(0, T, device=src_tokens.device).unsqueeze(0).expand(B, T)
        x = self.embed(src_tokens) + self.pos_embed(positions)
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        return self.norm(x)
