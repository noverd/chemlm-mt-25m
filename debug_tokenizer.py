import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from chemlm import ChemTokenizer

def debug_parsing():
    tokenizer = ChemTokenizer()
    test_smiles = "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C.NCCc1ccc(N)cc1F"
    tokenizer.build_vocab([(test_smiles, "C")])
    tokens = tokenizer.pattern.findall(test_smiles)
    indices = tokenizer.encode(test_smiles)

    print(f"Original: {test_smiles}")
    print(f"Tokens:   {tokens}")
    print(f"Indices:  {indices}")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    
    decoded = tokenizer.decode(indices)
    print(f"Decoded:  {decoded}")
    
    if test_smiles.replace(".", "") == decoded.replace(".", ""):
        print("\nCORRECT")
    else:
        print("\nINCORRECT")

if __name__ == "__main__":
    debug_parsing()
