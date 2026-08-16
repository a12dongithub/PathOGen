# Probe registry

Probe runs are identified by three independent IDs:

```text
classification task / model / counterfactual experiment / run ID
```

Keep task definitions in `tasks/`, encoder definitions in `models/`, and knob
definitions in `experiments/`. A run must copy the three selected definitions
into its manifest rather than encoding their meaning only in filenames.

The matching GCS layout is documented in `docs/artifact-layout.md`.
