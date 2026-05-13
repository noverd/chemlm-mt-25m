import sys
import os
import time
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from chemlm import ChemConfig, ChemSeq2Seq, ChemDataset, collate_fn, ChemTokenizer, load_mt_data


def get_device():
    if torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ckpt_dir, log_dir = "checkpoints", "logs"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_metrics.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("epoch,batch,loss,lr,time\n")

    device = get_device()
    raw = load_mt_data()
    if not raw: return
    tok = ChemTokenizer()

    tok.build_vocab(raw)
    conf = ChemConfig()
    conf.vocab_size = tok.vocab_size
    model = ChemSeq2Seq(conf).to(device)

    opt = optim.AdamW(model.parameters(), lr=7e-5, weight_decay=0.05)

    crit = nn.CrossEntropyLoss(ignore_index=tok.PAD, label_smoothing=0.1)
    loader = DataLoader(ChemDataset(raw, tok), batch_size=145, shuffle=True, collate_fn=collate_fn, num_workers=0)
    epochs = 10
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-4, total_steps=len(loader) * epochs)

    start_epoch = 0
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if ckpts:
        last = sorted(ckpts, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))[-1]
        checkpoint = torch.load(os.path.join(ckpt_dir, last), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            sched.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)

    model.train()
    for ep in range(start_epoch, epochs):
        start_t = time.time()
        pbar = tqdm(loader, desc=f"Epoch {ep + 1}/{epochs}")
        for b_idx, (src, tgt) in enumerate(pbar):
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad(set_to_none=True)
            dtype = torch.bfloat16 if device.type != 'cpu' else torch.float32
            with torch.autocast(device_type=device.type, enabled=(device.type != 'cpu'), dtype=dtype):
                out = model(src, tgt[:, :-1])
                loss = crit(out.reshape(-1, conf.vocab_size), tgt[:, 1:].reshape(-1))
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            sched.step()
            if b_idx % 2 == 0:
                with open(log_file, "a") as f:
                    f.write(
                        f"{ep + 1},{b_idx},{loss.item():.6f},{sched.get_last_lr()[0]:.8e},{time.time() - start_t:.2f}\n")
            if b_idx % 5 == 0: pbar.set_postfix({"loss": f"{loss.item():.3f}"})
            del out, loss
            if b_idx % 500 == 0 and device.type == "xpu": torch.xpu.empty_cache()

        save_path = os.path.join(ckpt_dir, f"chemlm_epoch_{ep + 1}.pt")
        torch.save({'epoch': ep + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': sched.state_dict(), 'config': conf, 'tokenizer_vocab': tok.vocab},
                   save_path)


if __name__ == "__main__":
    main()
