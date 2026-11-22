---
language: fr
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen3-8B
pipeline_tag: text-generation
tags:
  - medical
  - icd-10
  - clinical-coding
  - pmsi
  - agent
  - tool-use
---

# PARTAGES-CU2: LLM agent for French ICD-10 medical coding

Given the reports of a French hospital stay, this model codes the stay as in the PMSI (the French
hospital activity database): its **principal diagnosis (DP)** and, when the DP is a Z code, its
**related diagnosis (DR)**, in ICD-10 (CIM-10, ATIH version), with a **written justification**. It
is a Qwen3-8B fine-tuned to work as an agent: it searches the ICD-10 and the ATIH MCO
methodological guide with offline tools, follows a mandatory coding procedure, then answers. It
was developed in the PARTAGES project (use case 2, coding assistance for medical information
departments, DIM), next to the encoder
[`PARTAGES-CU2/cu2-icd10-dp-encoder`](https://huggingface.co/PARTAGES-CU2/cu2-icd10-dp-encoder).

**Terms used below**

- **DP** (*diagnostic principal*): the health problem that motivated the stay.
- **Z code**: an ICD-10 code of chapter XXI, which describes a reason for contact rather than a
  disease (e.g. `Z51.1` chemotherapy session, `Z09.0` follow-up after surgery).
- **DR** (*diagnostic relié*): when the DP is a Z code, the disease behind it (e.g. the cancer
  treated during a chemotherapy session).
- **Target code**: the disease of the stay, i.e. the DR if there is one, otherwise the DP.

**In short**

- **For:** coding assistance with an explanation the DIM can check. **Not for:** automatic coding
  without validation; no associated diagnoses (DAS).
- **Runs locally** with [vLLM](https://github.com/vllm-project/vllm) and offline tools: no text
  leaves your infrastructure. Trained without any real patient data.
- **Needs its harness:** the model only works with the tools, prompts and answer checks of the
  repository below (it was trained with them).
- **Hardware:** one GPU with ≥ 40 GB of memory recommended (A100, H100, L40S; tested on an A100
  80 GB): the weights take 16 GB and a long stay needs a 32 000-token context.
- **Version `v0.1`** (pre-release, to test the agent on your stays); `v0` is the previous one.

**Everything to run and evaluate it on your stays:**
<https://github.com/aphp-datascience/partages-cu2-decoder>.

## Quick use

```bash
git clone https://github.com/aphp-datascience/partages-cu2-decoder.git && cd partages-cu2-decoder && uv sync
uv run huggingface-cli download PARTAGES-CU2/cu2-icd10-dp-agent --local-dir models/cu2-icd10-dp-agent
uv run huggingface-cli download PARTAGES-CU2/cu2-agent-resources --repo-type dataset --local-dir resources/data
uv run python -m scripts.predict --model models/cu2-icd10-dp-agent --resources resources/data \
    --data my_stays.csv --out preds.csv
```

Input: the CSV of the PARTAGES guidelines, one row per stay: `id`, `speciality`, `text` (the
consultation, operative and discharge reports concatenated), `procedure` (the CCAM codes,
separated by spaces) and, for evaluation, `primary_diag` (the PMSI DP). Output per stay: the
suggested DP (`primary_diag_pred`), `dr`, `target` and the model's `justification`.

## Model and training

- **Base model:** `Qwen/Qwen3-8B` (Apache-2.0); LoRA fine-tuning (rank 32, all projections, one
  epoch, learning rate 1e-4), adapter merged into the weights (bf16).
- **Training data:** about 27 600 tool-using conversations of **Mistral Large 4** coding stays with
  the same tools and procedure, no real patient data:

  | Stays coded | Conversations | Kept |
  |---|---:|---|
  | [PARHAF](https://huggingface.co/datasets/HealthDataHub/PARHAF) (CC BY 4.0) fictitious patients and their rewrites | 6 385 | all |
  | PARHAF patients with a reference coding (Claude Opus, partly reviewed by physicians) | 784 | only conversations reaching the reference code |
  | MISTRAL synthetic surgical stays | 20 428 | all (Mistral Large 4's coding replaces the noisy original code) |

  One training example is one assistant turn of a conversation (tool call or final answer).
- **Prompt:** the system prompt includes the coding procedure the model was trained with,
  `procedure.md` in this repository; the inference script of the repository reads it from here.
- **Tools:** ICD-10 search (labels, inclusion notes, synonyms; BM25), code sheet (PMSI status,
  inclusions, exclusions), comparison of the codes of a category, search in the ATIH MCO guide 2026.

## Results (before any fine-tuning on real hospital data)

**Physician-reviewed AP-HP stays** (May 2026; text = acts + CRC + CRH + CR-OP), the same stays as for
the [encoder](https://huggingface.co/PARTAGES-CU2/cu2-icd10-dp-encoder). Views: **flattened** =
every review is a row; **agreement + single** = one row per stay, reviewed once or by several
reviewers who agree; **agreement** = only stays with several agreeing reviewers. Accuracy on the
disease of the stay (the DR if any, else the DP), whole ICD-10:

| View | Stays | Exact | 4 char. | 3 char. |
|---|---:|---:|---:|---:|
| flattened | 310 | 46.8 % | 48.7 % | 61.0 % |
| agreement + single | 195 | 47.2 % | 49.7 % | 62.1 % |
| agreement | 47 | 70.2 % | 70.2 % | 74.5 % |

**Principal diagnosis as coded (DP)**, what the repository's script compares with `primary_diag`
(flattened): 46.8 % exact, 48.5 % 4 char., 59.8 % 3 char.; 46.6 % when the DP is not a Z code and
47 % when it is a Z code (sessions, follow-up), where the agent is much better than the encoder
(about 25 %). Small sets: 95 % confidence interval about ±7 points on ~200 stays, ±13 points on ~50.

## Limitations

- Coding **assistance** only; the justification must be read critically.
- Slower than the encoder: several tool calls per stay.
- Mostly trained on surgical stays, with fictitious or synthetic records only.
- The codes used for training and evaluation are not certain, and some stays can legitimately be
  coded in more than one way.
- Evaluated on real stays of a single hospital group (AP-HP; AP-HP data used for evaluation only).
