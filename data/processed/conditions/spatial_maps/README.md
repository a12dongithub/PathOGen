# Five-channel spatial maps

Generate one `<tile-stem>.npz` file per annotated tile using
`python -m cpathogen.preprocessing.spatial_maps`. Each NPZ must contain key
`map` with shape `(512, 512, 5)` and channel order documented in `configs/data/`.
