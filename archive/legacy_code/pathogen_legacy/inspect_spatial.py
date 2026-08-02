import numpy as np
from pathlib import Path

d = Path('../results/512_final_dataset/spatial_maps')
files = sorted(d.glob('*.npz'))[:5]

for f in files:
    m = np.load(f)['map']
    print(f"{f.stem}:")
    print(f"  shape: {m.shape}")
    for c in range(m.shape[2]):
        nz = int((m[:,:,c] > 0).sum())
        mx = int(m[:,:,c].max())
        print(f"  ch{c}: max={mx}, nonzero_pixels={nz}")
    print()
