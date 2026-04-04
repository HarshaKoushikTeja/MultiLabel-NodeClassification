import scipy.io
import os

ppi = scipy.io.loadmat('data/processed/Protein_Protein_Interaction.mat')
wiki = scipy.io.loadmat('data/processed/Wikipedia.mat')

print("── PPI ────────────────────────────")
print(f"  network type  : {type(ppi['network'])}")
print(f"  network shape : {ppi['network'].shape}")
print(f"  group shape   : {ppi['group'].shape}")

print("\n── Wikipedia ──────────────────────")
print(f"  network type  : {type(wiki['network'])}")
print(f"  network shape : {wiki['network'].shape}")
print(f"  group shape   : {wiki['group'].shape}")