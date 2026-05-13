import torch
import torch.nn as nn
from .blocks import DecoderBlock

class ChemDecoder(nn.Module):
    def __init__(self, config, embed_layer, pos_embed_layer):
        super().__init__()
        self.embed = embed_layer
        self.pos_embed = pos_embed_layer
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, tgt_tokens, memory, tgt_mask=None, memory_padding_mask=None):
        B, T = tgt_tokens.size()
        positions = torch.arange(0, T, device=tgt_tokens.device).unsqueeze(0).expand(B, T)
        x = self.embed(tgt_tokens) + self.pos_embed(positions)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_padding_mask=memory_padding_mask)
        x = self.norm(x)
        logits = self.fc_out(x)
        return logits
