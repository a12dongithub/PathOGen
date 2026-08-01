# CPathoGen documentation

This directory explains the repository as it exists on 2026-07-26. It separates verified implementation details from poster/abstract claims and from future experiment plans.

## Recommended reading order

1. [Background for engineers](onboarding/background.md) - learn H&E, medical vocabulary, model families, datasets, metrics, and interpretation limits.
2. [Project overview](onboarding/project-overview.md) - understand the research question and project boundaries.
3. [Architecture](architecture/system.md) - understand preprocessing, training, inference, and probing.
4. [Repository map](reference/repository-map.md) - locate source, data, checkpoints, outputs, third-party code, and duplicate copies.
5. [Setup and workflows](guides/setup-and-workflows.md) - see what is required to run each part.
6. [Known issues](reference/known-issues.md) - review blockers and interpretation risks before changing code or citing results.
7. [Paper experiment registry](research/paper-experiments.md) - see the planned evaluation and publication program.

## Source-of-truth policy

Use the following priority when documents disagree:

1. Current executable code and stored configuration for implementation details.
2. Versioned history in the root `.git/`, migrated from the former nested generator repository, for generator evolution.
3. Result files for the exact run they describe.
4. Posters and abstracts for project motivation and preliminary claims.
5. Filenames, directory names, and archive names only as hints.

Result artifacts are not assumed to be reproducible unless their data split, checkpoint, preprocessing artifact, seed, and command can be recovered.

## Documentation ownership

| File | Update when |
|---|---|
| `README.md` at repository root | The project entry point, canonical paths, or headline status changes |
| `onboarding/background.md` | Medical terminology, model explanations, datasets, metrics, or onboarding guidance change |
| `onboarding/project-overview.md` | Research scope, datasets, model panel, or intended outputs change |
| `architecture/system.md` | Preprocessing, model components, tensor contracts, or inference changes |
| `reference/repository-map.md` | Files/folders are added, moved, archived, or declared canonical |
| `guides/setup-and-workflows.md` | Environments, commands, expected data layout, or run order changes |
| `reference/known-issues.md` | A blocker is discovered, resolved, or superseded |
| `research/paper-experiments.md` | Experimental priorities, metrics, baselines, or statistical protocol change |

## Terms used here

- **CPathoGen**: project/paper name used in the current abstract and posters.
- **PathOGen**: generator code repository and many internal filenames.
- **Phase 1**: H&E domain adaptation of the diffusion UNet with a constant `"he"` prompt.
- **Phase 2**: cellular spatial-map and morphology/stain conditioning.
- **Foundational experiments**: downstream studies that use generated counterfactuals to probe pathology encoders and classifiers.
- **Canonical**: the path to edit going forward; it does not imply that the code is packaged or fully reproducible.
