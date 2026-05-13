import torch
import os
from rdkit import Chem
from rdkit.Chem import Draw
from .models import ChemSeq2Seq, ChemConfig
from .data import ChemTokenizer

def get_best_device() -> str:
    if torch.xpu.is_available(): return "xpu"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

class ChemInference:
    def __init__(self, checkpoint_path: str, device: str | None = None) -> None:
        self.device = device if device else get_best_device()
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = ckpt['config']
        self.model = ChemSeq2Seq(self.config).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.tokenizer = ChemTokenizer()
        self.tokenizer.vocab = ckpt.get("tokenizer_vocab", self.tokenizer.vocab)
        self.tokenizer.char2idx = {ch: i for i, ch in enumerate(self.tokenizer.vocab)}
        self.tokenizer.idx2char = {i: ch for i, ch in enumerate(self.tokenizer.vocab)}
        self.tokenizer.vocab_size = len(self.tokenizer.vocab)

    @torch.no_grad()
    def predict(self, input_str: str, beam_size: int = 3, length_penalty: float = 1.0, repetition_penalty: float = 1.2) -> str:
        tokens = self.tokenizer.encode(input_str)
        src_tensor = torch.tensor([tokens + [self.tokenizer.EOS]], dtype=torch.long).to(self.device)
        memory = self.model.encoder(src_tensor)
        beams = [([self.tokenizer.SOS], 0.0, False)]
        for _ in range(self.config.max_seq_len):
            new_beams = []
            all_done = True
            for seq, score, done in beams:
                if done:
                    new_beams.append((seq, score, True)); continue
                all_done = False
                tgt_tensor = torch.tensor([seq], dtype=torch.long).to(self.device)
                logits = self.model.decoder(tgt_tensor, memory)
                l = logits[0, -1, :].clone()
                for t_id in set(seq):
                    if t_id > 3:
                        if l[t_id] < 0: l[t_id] *= repetition_penalty
                        else: l[t_id] /= repetition_penalty
                log_p = torch.log_softmax(l, dim=-1)
                top_v, top_i = torch.topk(log_p, beam_size)
                for i in range(beam_size):
                    nxt_t = top_i[i].item()
                    new_beams.append((seq + [nxt_t], score + top_v[i].item(), nxt_t == self.tokenizer.EOS))
            beams = sorted(new_beams, key=lambda b: b[1] / (len(b[0]) ** length_penalty), reverse=True)[:beam_size]
            if all_done: break
        return self.tokenizer.decode(beams[0][0])

    def draw_to_file(self, smiles: str, filename: str = "result.png") -> bool:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                Draw.MolToFile(mol, filename, size=(300, 300))
                return True
        except: pass
        return False
