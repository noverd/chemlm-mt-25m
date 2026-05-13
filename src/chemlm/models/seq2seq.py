import torch
import torch.nn as nn
from .encoder import ChemEncoder
from .decoder import ChemDecoder

# главная модель seq2seq
class ChemSeq2Seq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # общие эмбеддинги
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        self.encoder = ChemEncoder(config, self.token_embed, self.pos_embed)
        self.decoder = ChemDecoder(config, self.token_embed, self.pos_embed)
        
        # инициализируем веса по науке
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def make_mask(self, sz):
        # маска чтоб декодер не смотрел в будущее
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        mem = self.encoder(src)
        mask = self.make_mask(tgt.size(1)).to(tgt.device)
        return self.decoder(tgt, mem, tgt_mask=mask)
