# Workflow 07: control-consistency ranking

Workflow 07 consumes a Workflow 06 run. It annotates both real and generated
images with the same CellViT++ checkpoint, rebuilds spatial maps and raw
morphology features, fits a `StandardScaler` on baseline features only, and
uses generated-vs-baseline spatial RMSE plus standardized morphology MAE to
rank generated tiles. It writes `ranking.csv` and top-k manifests under
`<run-dir>/control_consistency/`.

For a multi-candidate Workflow 06 run, it additionally writes
`selected_candidates.csv`, `selected_manifest.json`, and `selected/`, choosing
the lowest control-score candidate independently for every source tile.

The ranking is a conditional-control diagnostic. It must not be used to replace
the all-sample FID/KID from Workflow 06; any FID/KID on a selected top-k subset
is a secondary, explicitly labeled analysis.
