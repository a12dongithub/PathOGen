# CellViT++ models

```text
cellvit_plus_plus/
└── cellvit_sam_h_x40_amp_001/
    └── model.pth
```

`model.pth` was moved from
`/Users/varangrai/Documents/cellpathogen/CellViT-SAM-H-x40-AMP-001.pth`.

| Property | Value |
|---|---|
| Architecture | `CellViTSAM` |
| Backbone | `SAM-H` |
| Resolution/magnification | x40 / 0.25 µm per pixel |
| Nucleus output classes | Background + five PanNuke classes |
| Training precision | AMP |
| SHA-256 | `356418f19d9d478f164c7a31f85274584fefaa02355815c09f52346c658c8ec4` |

The five non-background classes are Neoplastic, Inflammatory, Connective, Dead,
and Epithelial. The large checkpoint is ignored by ordinary Git; use an approved
model registry or Git LFS to distribute it. The tracked reference configuration
is `configs/models/cellvit_sam_h_x40_amp_001.yaml`.
