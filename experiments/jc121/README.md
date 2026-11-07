# Evaluating models on MIME

This evaluation measures action recognition on the two conditions shown in the MIME website leaderboard. The selected models receive identical ordered video frames and zero-shot prompts. Results and large artifacts live under `/mnt/efs/justin/0_experiments/jc121`.

## 1. Evaluation setup

The evaluation contains 86 actions in each of `base+blank@0` and `adversarial+misaligned@0`, with separate multiple-choice and free-form requests. The original six-model run contains 2,064 answers. Revision `jc121.1` adds 344 GPT-6 Astra answers and preserves the original observations, producing a seven-model comparison with 2,408 answers.

| Component | Setting |
| --- | --- |
| Models | GPT-6 Astra, GPT-6.1 Sol, Claude Opus 5.5, DeepSeek V4.1 Flash 1017, Qwen3.8 Flash, GLM 5.3 Flash, MiniMax M3.1 Flash |
| Data | `wise-east/mime-cropped`, revision recorded in `contract.yaml` |
| Frames | Every thirtieth source frame, starting at zero-based frame 29, resized within 512 × 512 pixels |
| Prompts | Released zero-shot prompts from `src/MimeEval/models/vlms.py` |
| Choices | Archived GPT-4o Mini zero-shot question sets, reshuffled once per action and shared across models |
| Reasoning | Medium requested; Qwen uses a 4,096-token thinking budget. All models have at most 8,192 total completion tokens |
| Multiple-choice score | Share of correct choices, with invalid answers counted as incorrect |
| Free-form score | Share with MiniLM cosine similarity at least 0.5 to any accepted reference label |
| Uncertainty | Wilson 95% intervals for the share of correct answers in each 86-action condition |

Both accuracy metrics run from 0% to 100%; higher is better. The free-form cutoff follows the paper. The released scoring command's default cutoff differs. Pipe-separated reference labels are scored separately, taking the highest similarity. The sentence encoder runs through its published ONNX weights with attention-masked mean pooling and normalization, without local GPU inference.

## 2. Running the evaluation

Install the pinned packages in `requirements.txt` into an EFS-backed environment. Follow the shared memory-safety rules and launch each stage inside a verified memory limit. The artifact directory must already contain the frozen dataset metadata in `dataset_info.json`.

```bash
/usr/local/bin/intology-secrets-run python experiments/jc121/evaluate.py predict --output-dir /mnt/efs/justin/0_experiments/jc121/revisions/jc121.1 --models openai/gpt-6-astra --workers 8
python experiments/jc121/evaluate.py score --output-dir /mnt/efs/justin/0_experiments/jc121/revisions/jc121.1
python experiments/jc121/evaluate.py verify --output-dir /mnt/efs/justin/0_experiments/jc121/revisions/jc121.1
```

`predict` reads `ZOCHI_KEY` from its environment and sends requests through the Zochi gateway's OpenRouter route. The credential stays in memory. The gateway scope and local accounting share a $60 working limit. The local scheduler reserves $0.50 per active request before dispatch, and actual costs come from OpenRouter's response usage. The gateway's accounting can lag active requests.

The revision directory links to the frozen `cases`, `cases.json`, metadata, scorer, validation archives, and six model-response directories in the parent run. Astra responses and all derived comparison files belong to the revision directory. `--models` selects the models for prediction, scoring, and verification. The parent run can be rescored and verified by explicitly selecting its six models and leaving the output directory at its default.

The pilot uses `prepare --actions 5`, then `predict --actions 1`, producing answers for one action in both conditions. It checks operation and cost; it does not establish a model ranking. `score --partial` allows inspecting incomplete output during preparation. Only a successful full `verify` supports the complete evaluation report.

## 3. Reviewing the evidence

`cases/<sample_id>/case.json` joins the dataset row to its video, sampled frames, frame hashes, choices, and correct answer. `responses/<model>/<sample_id>/` contains the exact model-visible request and raw response for each answer format. Free-form requests contain no reference label. Source filenames are absent from the model-visible image content.

`predict` reuses complete saved responses when resumed. It preserves HTTP failures and retries only rate limits or errors marked safe to retry by the gateway, including nested error flags. Incomplete saved responses stop execution for inspection. `scores.json` preserves individual scores; `summary.json` contains per-condition aggregates. `verify` checks expected coverage, request and frame provenance, model identities, complete responses, score counts, and reconciled model charges before writing `integrity-report.json`.

The first route probe used the same frozen MIME request without a gateway scope. Its response is preserved in `route-check.json` and in `pilot-mcq-responses`. The current-metadata pilot questions differ from the archived paper questions, so pilot multiple-choice answers are excluded from final accuracy and charged separately as validation. Unchanged pilot free-form answers remain valid except Qwen outputs, which are rerun under the explicit thinking budget. `qwen-medium-responses` preserves those validation charges. Gateway-scope totals exclude the first unscoped probe.

## 4. Auditing free-form scoring

`audit_freeform.py` sends frozen reference-answer pairs to GPT-6.1 Sol and Claude Opus 5.5 for independent semantic judgments. It withholds the original model, condition, and embedding score. Each judge labels a pair as matching, mismatching, or ambiguous. These judgments audit the embedding metric; they are not human ground truth and do not replace the published scores.

The audit artifacts live in `/mnt/efs/justin/0_experiments/jc121/studies/freeform-audit`. `blind-pairs.json` contains the judge-visible pairs, `private-mapping.json` maps pairs to all original answers, and `judges/` preserves requests and responses. Run the script through `intology-secrets-run` inside a verified memory limit. It resumes complete batches and enforces a $5 additional API budget within the existing gateway scope.

`judge_jev.py` independently grades those reference-answer pairs with Jev 1.14 through the gateway's `/openrouter/systemone` route. It sends reference alternatives and the candidate answer, without audit labels or model identities. A fixed probability threshold of 0.5 determines credit. The artifacts in `studies/jev-judge` preserve each request, response, and the frozen comparison contract. The primary comparison excludes ambiguous audit labels; the model-level tables also compare the resulting accuracies with the audit-adjusted scores. This comparison measures agreement with an AI-assisted audit, not human ground truth.

## 5. Comparing open-source graders

`judge_open.py` evaluates released Verdict, Laya, Kev-4B, and Perplexity Decider Flash checkpoints on the same 840 blinded reference-answer pairs. It preserves the Jev rubric and 0.5 threshold, records native responses, checks for input truncation, and resumes completed pairs. Verdict uses its native binary choice and abstention interface. The other judges use their yes/no interfaces. No audit labels enter model inference, training, or threshold selection.

Artifacts live in `/mnt/efs/justin/0_experiments/jc121/studies/open-judges`. The contract and `models.json` pin the input hashes, checkpoint revisions, adapter behavior, and CPU execution. `setup_open_judges.sh` and `run_open_judges.sh` operate on the existing evaluation host inside a separately enforced memory boundary. They allocate no GPUs and make no paid API calls.

The larger judges select CPU thread count using repeated timing checks without audit labels. Their native batch paths must reproduce single-answer decisions on the execution check before the full sweep starts. Responses preserve the actual thread count and batch size, and `timing.json` measures repeated amortized time per answer. The memory ceiling remains 64 GiB throughout.

`analyze_open_judges.py --root <artifact-directory>` verifies coverage and input provenance, compares all graders with the definite audit labels, and writes `report.json`, `model-scores.csv`, and `disagreements.csv`. It also reports the average absolute gap from the fourteen audit-adjusted model-condition scores. Ambiguous audit cases remain excluded from primary agreement; their original embedding decisions remain in the secondary adjusted-score comparison. `--partial` is for monitoring only and does not produce a passing final integrity report.

Perplexity Decider Flash v1.2 is the recommended open-weight replacement on this sample. The table measures agreement with 1,127 definite AI-assisted audit labels, from 0% to 100%; higher is better. False credits and false rejections count disagreements with those labels; lower is better. The 77 ambiguous answers are excluded. All four open judges graded the same 840 distinct reference-answer pairs, covering 1,204 original answers.

| Grader | Audit agreement | False credits | False rejections |
| --- | --- | --- | --- |
| Jev 1.14 | 99.7% | 1 | 2 |
| Perplexity Decider Flash v1.2 | 99.6% | 4 | 0 |
| Kev-4B | 97.9% | 2 | 22 |
| Sentence embeddings | 95.4% | 45 | 7 |
| Laya | 87.5% | 87 | 54 |
| OpenJev Verdict 151M | 74.7% | 84 | 201 |

Perplexity and Jev each have a mean absolute gap of 0.75 percentage points from the fourteen audit-adjusted model-condition accuracies, compared with 3.49 points for embeddings; lower is better. This secondary target retains embedding decisions for ambiguous answers. These are agreements with a frozen AI-assisted audit, not independently human-labeled accuracy or proof of generalization. The recommendation concerns the grader; the repository's published embedding scores remain unchanged.
