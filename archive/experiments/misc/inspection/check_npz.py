import numpy as np

path = r"data/processed/conditions/spatial_maps/TCGA-A1-A0SF_x12288_y50176_BL.npz"
data = np.load(path)['map']

print("Channel 0 (Tumor): max =", data[:, :, 0].max())
print("Channel 1 (Immune): max =", data[:, :, 1].max())
print("Channel 2 (Stroma): max =", data[:, :, 2].max())
print("Channel 3 (Empty/Necrotic?): max =", data[:, :, 3].max())
print("Channel 4 (Epithelial?): max =", data[:, :, 4].max())
