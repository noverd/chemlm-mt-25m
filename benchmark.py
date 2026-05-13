import torch
import sys
import os
import re
import json
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from dataclasses import dataclass
from difflib import SequenceMatcher

RDLogger.DisableLog('rdApp.*')

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from chemlm import ChemInference

@dataclass
class EvalMode:
    name: str
    beam_size: int = 1
    length_penalty: float = 1.0
    repetition_penalty: float = 1.2
    do_sample: bool = False

MODES = [
    EvalMode("Strict", beam_size=3, length_penalty=0.0, repetition_penalty=1.0),
    EvalMode("Improved", beam_size=5, length_penalty=0.8, repetition_penalty=1.2),
]

def str_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def get_metrics(pred_s: str, target_s: str):
    text_sim = str_sim(pred_s, target_s)
    m1 = Chem.MolFromSmiles(pred_s)
    m2 = Chem.MolFromSmiles(target_s)
    if not m1: return False, 0.0, False, text_sim
    c1 = Chem.MolToSmiles(m1, canonical=True)
    c2 = Chem.MolToSmiles(m2, canonical=True)
    match = (c1 == c2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
    tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
    return True, tanimoto, match, text_sim

def run_bench():
    ckpt_dir, bench_dir, out_dir = "checkpoints", "benchmarks", "benchmark_results"
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(bench_dir) if f.endswith(".json")]
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")], 
                   key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))

    for cp_file in ckpts:
        ep_num = re.search(r'epoch_(\d+)', cp_file).group(1)
        engine = ChemInference(os.path.join(ckpt_dir, cp_file))
        all_results = []
        for f in files:
            cat = f.replace(".json", "").upper()
            with open(os.path.join(bench_dir, f), "r") as j:
                tests = json.load(j)
            print(f"\n--- Epoch {ep_num} | Category {cat} ---")
            print(f"| {'Test':<20} | Val | Tanimoto | Match | TextSim | {'Result':<25} |")
            print("-" * 95)
            for t in tests:
                for mode in MODES:
                    res = engine.predict(t["input"], beam_size=mode.beam_size, 
                                       length_penalty=mode.length_penalty,
                                       repetition_penalty=mode.repetition_penalty)
                    is_v, tan, m, t_sim = get_metrics(res, t["target"])
                    all_results.append({
                        "epoch": ep_num, "category": cat, "test_name": t["name"],
                        "mode": mode.name, "valid": is_v, "tanimoto": tan,
                        "exact_match": m, "text_similarity": t_sim,
                        "prediction": res, "target": t["target"]
                    })
                    if mode.name == "Improved":
                        v_i = "V" if is_v else "I"
                        m_i = "Y" if m else "N"
                        print(f"| {t['name']:<20} | {v_i:^3} | {tan:>8.1%} | {m_i:^5} | {t_sim:>7.1%} | {res[:25]:<25} |")
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(out_dir, f"results_epoch_{ep_num}.csv"), index=False)

if __name__ == "__main__":
    run_bench()
