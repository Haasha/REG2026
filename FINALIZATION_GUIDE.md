# REG2026 Grand Challenge Finalization Guide

This repository now contains integrated evaluation logic in all three phase kits.

## What Was Finalized

The following evaluation entrypoints were replaced with production logic:

- `debug-phase-kit/example_evaluation_method/evaluate.py`
- `test-phase-1-kit/example_evaluation_method/evaluate.py`
- `test-phase-2-kit/example_evaluation_method/evaluate.py`

Each evaluator now:

1. Reads `/input/predictions.json`
2. Processes both interfaces:
   - `interf0` (visual-context-question + thumbnail)
   - `interf1` (whole-slide-image + chain-of-thought)
3. Auto-discovers required ground truth assets under `/opt/ml/input/data/ground_truth`
4. Computes official outputs:
   - `workflow_final_ranking_score`
   - `visual_final_score`
5. Writes `/output/metrics.json`

## Required Ground Truth Tarball Contents

Your uploaded `ground_truth.tar.gz` must include:

1. One workflow JSON file containing cases with:
   - `id`
   - one of: `chain-of-thought`, `chain_of_thought`, `workflow`, `workflow_steps`, `steps`
2. One tab-separated mapping TXT containing at least these columns:
   - `anonymous_id`
   - `image`
   - `question`
   - `variant`
   - `label`
   - `paired_anonymous_id`
   - `b3_paired_anonymous_id`

You can keep additional files in the same tarball.

## Output Format

`/output/metrics.json` includes:

- `aggregates.workflow_final_ranking_score`
- `aggregates.visual_final_score`
- `official_scores.workflow_final_ranking_score`
- `official_scores.visual_final_score`
- detailed per-case and per-interface diagnostics under `details`

## Packaging Per Phase

For each phase kit:

1. Place the correct ground truth assets in `example_evaluation_method/ground_truth`
2. Build and test locally:
   - `./do_test_run.sh`
3. Save for upload:
   - `./do_save.sh`
4. Upload to Grand Challenge:
   - evaluation image tar.gz
   - `ground_truth.tar.gz`

## Important Notes

- The evaluator is now platform-safe and does not depend on external local LLM paths.
- Workflow semantic comparison uses lexical similarity.
- Visual metric uses deterministic text-based judging aligned with your scripts-final structure.

If you want stricter or LLM-based judging later, add that in a separate branch and keep this version as the stable upload baseline.
