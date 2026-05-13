import re
import torch
from typing import Union

class ChemTokenizer:
    def __init__(self) -> None:
        self.PAD: int = 0
        self.SOS: int = 1
        self.EOS: int = 2
        self.UNK: int = 3
        
        # Спецтокены для текущих задач
        self.TASK_SYN: str = "[SYNTHESIS]"
        self.TASK_RETRO: str = "[RETRO]"

        self.vocab: list[str] = [
            '<PAD>', '<SOS>', '<EOS>', '<UNK>',
            self.TASK_SYN, self.TASK_RETRO
        ]
        self.char2idx: dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx2char: dict[int, str] = {i: ch for i, ch in enumerate(self.vocab)}
        
        # Регулярное выражение для SMILES + спецтокены
        self.pattern: re.Pattern = re.compile(
            r"(\[SYNTHESIS\]|\[RETRO\]|\[[^\]]+\]|Br?|Cl?|[a-zA-Z]|[0-9]|[^a-zA-Z0-9\s])"
        )
        self.vocab_size: int = len(self.vocab)

    def build_vocab(self, raw_data: list[tuple[str, str]]) -> None:
        print("Собираем словарь из датасета...")
        unique_tokens: set[str] = set()
        for s1, s2 in raw_data:
            tokens1: list[str] = self.pattern.findall(s1)
            tokens2: list[str] = self.pattern.findall(s2)
            unique_tokens.update(tokens1 + tokens2)
        
        for token in sorted(list(unique_tokens)):
            if token not in self.char2idx:
                idx: int = len(self.vocab)
                self.vocab.append(token)
                self.char2idx[token] = idx
                self.idx2char[idx] = token
        
        self.vocab_size = len(self.vocab)
        print(f"Словарь собран! Размер: {self.vocab_size} токенов.")

    def encode(self, text: str) -> list[int]:
        tokens: list[str] = self.pattern.findall(text)
        return [self.char2idx.get(token, self.UNK) for token in tokens]

    def decode(self, indices: list[int] | torch.Tensor) -> str:
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
            
        return "".join([
            self.idx2char.get(idx, '<UNK>') 
            for idx in indices 
            if idx not in (self.PAD, self.SOS, self.EOS)
        ])
