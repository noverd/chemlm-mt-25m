import sys
import os
import torch
from rdkit import Chem
from re import search

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from chemlm import ChemInference


def main():
    ckpt_dir = "checkpoints"
    if not os.path.exists(ckpt_dir):
        print("error: checkpoints directory not found")
        return

    files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
                   key=lambda x: int(search(r'epoch_(\d+)', x).group(1)) if search(r'epoch_(\d+)', x) else 0)

    if not files:
        print("error: no weights found")
        return

    print("\navailable epochs:")
    for i, f in enumerate(files):
        print(f"{i + 1}. {f}")

    try:
        choice = int(input("\nchoose epoch number (or press enter for latest): ") or len(files))
        selected_file = files[choice - 1]
    except:
        selected_file = files[-1]

    path = os.path.join(ckpt_dir, selected_file)
    print(f"loading: {path}")

    engine = ChemInference(path)

    print("\nChemLLM ready!")

    while True:
        print("\n1. [SYNTHESIS]\n2. [RETRO]\n0. exit")
        c = input("> ").strip()
        if c == '0': break
        pref = "[SYNTHESIS]" if c == '1' else "[RETRO]" if c == '2' else ""
        if not pref: continue

        inp = input("SMILES: ").strip()
        if not inp: continue

        res = engine.predict(pref + inp)
        mol = Chem.MolFromSmiles(res)
        print(f"\nResult: {res}\nStatus: {'Valid' if mol else 'Invalid'}")
        if mol:
            engine.draw_to_file(res, "last_prediction.png")
            print("image saved to last_prediction.png")


if __name__ == "__main__":
    main()
