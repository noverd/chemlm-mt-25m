from dataclasses import dataclass

@dataclass
class ChemConfig:
    vocab_size: int = 550
    max_seq_len: int = 128     
    d_model: int = 384
    n_heads: int = 8           
    n_layers: int = 6
    dropout: float = 0.1
    feedforward_dim: int = 1536
