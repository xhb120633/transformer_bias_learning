# Behavioral model ablations: code and figure inputs

This repository accompanies the independent-test preprint. Its scientific
files match the accompanying `preprint_code.zip`; the ZIP and repository are
two distributions of the same code and frozen main-figure inputs. The larger
synthetic data, evaluation exports, and trained models are stored in the
[data and model repository](https://huggingface.co/xhb120633/behavioral-model-ablations).

## What is where

- Preprint code ZIP: source, configurations, requirements, tests, frozen test
  summaries, figure scripts, and the exact manuscript figure PDFs.
- Data and model repository: https://huggingface.co/xhb120633/behavioral-model-ablations
  `review_artifacts/` contains the code plus full synthetic test records,
  prepared test prompts, participant-level probability exports, cognitive fits,
  audit manifests, and the previously released train/validation data and 168
  validation-selected GRU/Transformer checkpoints.
- `task_a/` and `task_b/` in that repository contain all 28 final LLaMA adapters.
  Base LLaMA weights are not redistributed. Use the specified base model under
  its applicable access and licensing terms.

## Quick checks and main-figure rebuild (no inference)

From the package root, with Python 3.10 or newer:

```sh
python -m pip install -r requirements.txt
python -m pytest -q -p no:cacheprovider tests/test_four_armed_restless.py tests/test_spatial_mixture.py tests/test_reward_ablation.py
python scripts/reproduce_figures.py
```

The last command renders Figures 1 to 4 and the choice-only behavioral-baseline
supplement (Figure S3 in the manuscript). It uses the independent-test summaries
in `archive/outputs/test_main_integrated_20260925/`, not old validation curves.
Figure 1A retains the manuscript's labeled illustrative validation trial;
Figure 1C uses test means. `paper_figures/` contains the exact frozen PDFs for
all main and supplementary figures. Other supplementary generator/operation
analyses remain validation or training analyses, as labeled in the manuscript.
Saved-summary plotting is not a rerun of model training.

## Full test reaggregation (download review_artifacts first)

Run from the root containing `src/`, `data/`, `outputs/`, and `scripts/`:

```sh
python scripts/collect_llama_test_20260925.py --self-test
python scripts/collect_llama_test_20260925.py --task both --output reproduced_test_summaries
```

This uses the released participant exports and checks frozen checkpoint/script
provenance, subject IDs, actual choices, donor mappings, conditional probability
normalization, causal probes and metric identities before aggregation. It does
not run LLaMA inference. Original absolute compute paths have been replaced
with package-relative paths. Three resumed shards use anonymized extracts of
the original successful causal-probe logs; their source hashes are retained in
`ANONYMIZATION.json`. The release inventory hashes the sanitized files.

## Experimental routes and GPU requirements

| Paper component | Code |
| --- | --- |
| Restless and spatial generators | `src/mechcal/generators/`, `scripts/run_weight_curve_a.py`, `scripts/generate_spatial_pilot.py` |
| Scratch neural training | `scripts/run_weight_neural_suite.py`, `scripts/run_spatial_neural.py` |
| Non-LLM test evaluation and reference fitting | `scripts/test_nonllm_20260925.py`, `scripts/test_nonllm_audit_20260925.py` |
| Frozen LLaMA test payloads | `data/llama_test_20260925/`, `scripts/prepare_llama_test_20260925.py` |
| LLaMA test inference | `scripts/eval_weight_llama.py --test-main`, `scripts/eval_spatial_llama.py --test-main` |
| Test collection | `scripts/collect_llama_test_20260925.py` |
| Main-figure reconstruction | `scripts/reproduce_figures.py` |

Use each script's `--help` for task, mode and weight arguments. The non-LLM
pipeline phases are prepare, neural, spatial_fits, references, and aggregate;
the released caches allow aggregation without rerunning fits. Never overwrite
the archived test results when exploring new settings. Models retain their
validation-selected checkpoints, with no test-based tuning or retraining.
Individual fitted generators use only the first 150 trials or first six maps;
scored targets are trials 151-200 or maps 7-8. Both tasks use seven weights and
20 shared donor derangements, with actual choices retained.

The requirements file supports the analysis and scratch-network paths, not a
locked 70B GPU environment. LLaMA additionally needs compatible Unsloth, TRL,
Transformers, bitsandbytes, CUDA, and substantial GPU memory. To rerun inference,
place adapters in the relative paths recorded in `FROZEN_INPUTS.json`; their
Hub locations are `task_a/{full,choice_only}/reward_wXXX` and
`task_b/{full,choice_only}/reward_wXXX`.
No full retraining or 70B inference rerun is claimed for this release check.

`MANIFEST.json` gives SHA-256 hashes. `RELEASE_SCOPE.json` records which materials
are test versus development. The manuscript is distributed separately. No
code or data license is implied by this snapshot.
