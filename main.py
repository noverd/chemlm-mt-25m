import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from chemlm import ChemConfig, ChemSeq2Seq

def test_run():
    config = ChemConfig()
    model = ChemSeq2Seq(config)

    print(f"Параметров в ChemLLM: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    dummy_src = torch.randint(0, config.vocab_size, (2, 15))
    dummy_tgt = torch.randint(0, config.vocab_size, (2, 10))

    logits = model(dummy_src, dummy_tgt)

    print(f"Форма входа реагентов: {dummy_src.shape}")
    print(f"Форма входа продуктов: {dummy_tgt.shape}")
    print(f"Форма выхода (логитов): {logits.shape}")

if __name__ == "__main__":
    test_run()
