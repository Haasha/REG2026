# REG2026 Grand Challenge Organizer Playbook

This document is for the organizer side of running REG2026 on Grand Challenge.

It answers four practical questions:

1. What in the starter kit is only an example and must be replaced?
2. What must you upload and configure yourself on the platform?
3. What problems are platform-side and should be discussed with the Grand Challenge team?
4. What is the recommended order for finalizing Debug, Test Phase 1, and Test Phase 2?

## 1. What The Current Platform Messages Mean

From your screenshots/text, the current phase status means:

- `0 valid archive items`: no usable cases are currently present in the linked archive for that phase.
- `0 algorithm jobs will be created`: because the linked archive has no valid items yet.
- `no valid evaluation method`: there is no successfully uploaded and activated evaluation container for that phase.
- `score jsonpath is not set`: the evaluation may run after upload, but nothing will appear properly on the leaderboard until scoring is configured in Phase Settings.
- `challenge has exceeded its budget`: this is not a code problem. This requires Grand Challenge support/admin action.

So at the moment, submissions are blocked for three separate reasons:

1. no active evaluation method
2. no scoring configuration
3. budget issue on the platform

## 2. What In The Starter Kit Is Just Template Code

The starter kit is not production-ready by default. It is a scaffold.

### Evaluation Method

These files are organizer-owned and must be finalized by you:

- `debug-phase-kit/example_evaluation_method/evaluate.py`
- `test-phase-1-kit/example_evaluation_method/evaluate.py`
- `test-phase-2-kit/example_evaluation_method/evaluate.py`

These were template files originally. They now contain integrated evaluation logic in this repo, but they still depend on you providing the correct ground-truth assets and correct platform configuration.

You also own:

- `requirements.txt`
- `Dockerfile`
- `do_test_run.sh`
- `do_save.sh`
- contents of `ground_truth/`

### Submission Algorithm Example

These algorithm files are examples, not your participants' real submissions:

- `example_algorithm/inference.py`
- `example_algorithm/model/`

You do not have to build the participants' algorithm for them. But you should keep one working example algorithm or smoke-test algorithm so you can verify the phase end-to-end.

If the current example algorithm only writes placeholder outputs, it is suitable only for container plumbing tests, not for realistic challenge validation.

### Archive Upload Script

These are also organizer-owned examples:

- `upload_to_archive/upload_files.py`

This script is only useful if you want to upload archive items programmatically. You must replace placeholder token usage and expected local files if you intend to use it.

## 3. What You Must Replace Yourself

For each phase, you must provide real content in the following areas.

### A. Real Ground Truth

For each phase, the uploaded `ground_truth.tar.gz` should contain the actual evaluation assets, not starter placeholders.

For the current integrated evaluator, each phase ground truth needs at minimum:

1. A workflow ground-truth JSON file
2. A visual mapping TXT file

Expected workflow JSON structure:

- top-level list of cases, or dict containing `cases`
- each case needs `id`
- each case needs one of:
  - `chain-of-thought`
  - `chain_of_thought`
  - `workflow`
  - `workflow_steps`
  - `steps`

Expected mapping TXT columns:

- `anonymous_id`
- `image`
- `question`
- `variant`
- `label`
- `paired_anonymous_id`
- `b3_paired_anonymous_id`

The current evaluator auto-discovers these inside `/opt/ml/input/data/ground_truth/`.

### B. Real Archive Items

Each linked archive must contain valid cases for the phase.

Your configured interfaces are:

#### Interface 1

- Visual context question (String)
- Histopathology region of interest thumbnail (Thumbnail jpg)

Each archive item for Interface 1 must include both of those values in a single archive item.

#### Interface 2

- Whole Slide Image (Image)

Each archive item for Interface 2 must include one whole slide image.

If you upload only one piece of data without matching the configured socket combination, the item will not count as valid for that interface.

### C. Real Scoring Configuration In Phase Settings

This is a platform configuration step, not a code step.

Your evaluator writes these paths into `/output/metrics.json`:

- `official_scores.workflow_final_ranking_score`
- `official_scores.visual_final_score`
- `aggregates.overall_two_metric_mean`

You must decide what the main leaderboard ranking score should be.

Recommended options:

1. If workflow score is the official ranking metric for the phase:
   - Score title: `Workflow Final Ranking Score`
   - Score jsonpath: `official_scores.workflow_final_ranking_score`
   - Extra result column: `official_scores.visual_final_score`

2. If visual score is the official ranking metric for the phase:
   - Score title: `Visual Final Score`
   - Score jsonpath: `official_scores.visual_final_score`
   - Extra result column: `official_scores.workflow_final_ranking_score`

3. If the leaderboard should rank by a combined metric:
   - Score title: `Overall Two Metric Mean`
   - Score jsonpath: `aggregates.overall_two_metric_mean`
   - Extra result columns:
     - `official_scores.workflow_final_ranking_score`
     - `official_scores.visual_final_score`

You need to make that ranking decision explicitly as organizer.

## 4. What You Need To Upload Yourself

For each phase, you should upload the following.

### Required uploads

1. Evaluation container image tar.gz
2. Ground truth tar.gz
3. Archive items in the linked archive

### Optional but strongly recommended

1. One working example algorithm container
2. One small smoke-test dataset for internal verification

## 5. What Needs To Be Configured On The Website By You

For each of Debug, Test Phase 1, and Test Phase 2:

### Evaluation Methods tab

- Upload the saved evaluation method container tar.gz
- Wait for validation to complete
- Confirm it becomes the active method

### Ground Truths tab

- Upload `ground_truth.tar.gz`
- Confirm the phase accepts it

### Linked Archive tab

- Upload valid archive items manually or via API/script
- Confirm the valid item count is non-zero
- Confirm the algorithm jobs count becomes non-zero

### Phase Settings -> Scoring

- Set Score title
- Set Score jsonpath
- Add extra result columns if you want multiple displayed metrics
- Set scoring order direction correctly

### Phase Settings -> Submission

- Open/close dates
- submission limits
- participant requirements
- instructions shown to participants

### Phase Settings -> Leaderboard / Result Details

- choose what is displayed publicly
- decide whether to show full metrics output or a clean summary

## 6. What To Discuss With Grand Challenge Support

These are not things you should spend time debugging in starter code.

### Must discuss with support

1. Budget exceeded
   - Your platform currently says submissions are blocked because the challenge exceeded its budget.
   - This must be resolved by Grand Challenge staff.

2. Runtime requirements if you need more than standard evaluation resources
   - evaluation GPU type
   - memory limit
   - any non-standard compute/storage needs

3. Algorithm runtime configuration for participant submissions
   - if you need specific GPU/CPU/memory for participant algorithms
   - if your WSI jobs need special viewer/image handling

4. Any uncertainty in how multiple official metrics should be represented on the leaderboard
   - single ranking metric plus extra columns is standard
   - if you need multiple leaderboards or a non-standard ranking policy, confirm with them

5. Any archive/socket validation behavior that does not match your intended data model
   - especially if archive items appear uploaded but remain invalid

### Good questions to send them

- `Our phases are blocked by budget exceeded. Can you re-enable budget or advise the next step?`
- `For REG2026, we have two evaluation outputs. Do you recommend one primary score jsonpath with the second metric as an extra leaderboard column, or a multi-leaderboard setup?`
- `What evaluation runtime limits are currently configured for our phases, and are they suitable for WSI-scale evaluation?`
- `Can you confirm our current archive/socket configuration is correct for the two interfaces before we bulk upload all hidden test data?`

## 7. What You Should Not Change Right Now

Avoid changing these until the platform basics are working:

- do not redesign all metrics again before a first successful upload
- do not optimize for final scientific strictness before a first successful end-to-end phase run
- do not spend time debugging participant algorithm behavior before the evaluation method, ground truth, scoring, and archive are all active

Your first milestone is operational correctness, not final research polish.

## 8. Recommended Finalization Order

Do this first for Debug Phase, then repeat for Test Phase 1 and Test Phase 2.

### Step 1. Finalize the evaluation package locally

- put real ground truth files into the phase `ground_truth/` folder
- run local test scripts on a Docker-enabled machine
- export evaluation image with `do_save.sh`

### Step 2. Upload and activate evaluation method

- upload the container in Evaluation Methods
- verify it becomes valid and active

### Step 3. Upload ground truth tarball

- upload `ground_truth.tar.gz`
- verify platform accepts it

### Step 4. Configure scoring

- set the main score jsonpath
- add extra displayed metrics if needed

### Step 5. Upload a very small set of valid archive items

- confirm the valid item count is correct
- confirm algorithm job count is non-zero

### Step 6. Submit a smoke-test algorithm

- use a simple algorithm container
- verify evaluation runs and produces metrics

### Step 7. Only then upload the full hidden archive

- bulk upload or script upload after the schema is confirmed correct

## 9. Practical Split: Your Work vs Support Work

### Your work

- finalize evaluator code
- prepare real ground-truth assets
- prepare archive items
- upload evaluation image
- upload ground truth tarball
- configure scoring jsonpaths
- configure submission rules
- run smoke tests per phase

### Grand Challenge support work

- resolve budget block
- confirm or adjust runtime resources
- assist with non-standard leaderboard/scoring setup if needed
- assist if socket/archive configuration on the platform needs change

## 10. Current Status Of This Repository

This repo now has integrated evaluation code for all three phase kits.

Still required from you before the platform becomes operational:

1. replace placeholder ground truth content with real phase assets
2. upload evaluation method containers
3. upload ground truth tarballs
4. upload valid archive items
5. configure score jsonpaths
6. contact support about the budget block

## 11. Minimum Success Criteria Per Phase

You can consider a phase operational when all of the following are true:

1. Linked archive shows non-zero valid items
2. Algorithm jobs created is non-zero
3. Evaluation Methods shows a valid active method
4. Ground Truth is uploaded successfully
5. Phase Settings has score jsonpath configured
6. One test submission runs to completion
7. Leaderboard displays the intended metric columns