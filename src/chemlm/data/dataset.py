import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# класс для работы с данными
class ChemDataset(Dataset):
    def __init__(self, data_list, tokenizer):
        self.data = data_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # берем одну пару реакция-продукт
        src_raw, tgt_raw = self.data[idx]
        
        # кодируем в числа
        src = torch.tensor(self.tokenizer.encode(src_raw), dtype=torch.long)
        tgt = torch.tensor(self.tokenizer.encode(tgt_raw), dtype=torch.long)
        return src, tgt

# склеивание батча
def collate_fn(batch):
    src_list, tgt_list = [], []
    for src, tgt in batch:
        src_full = torch.cat([src, torch.tensor([2])]) # 2 - это EOS
        tgt_full = torch.cat([torch.tensor([1]), tgt, torch.tensor([2])]) # 1 - SOS
        
        src_list.append(src_full)
        tgt_list.append(tgt_full)
        
    # добиваем нулями до одной длины
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=0)
    return src_padded, tgt_padded
