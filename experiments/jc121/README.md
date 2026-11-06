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
