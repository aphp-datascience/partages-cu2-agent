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

**In short**

- **For:** coding assistance with an explanation the DIM can check. **Not for:** automatic coding
  without validation; no associated diagnoses (DAS).
- **Runs locally** with [vLLM](https://github.com/vllm-project/vllm) and offline tools: no text
  leaves your infrastructure. Trained without any real patient data.
- **Needs its harness:** the model only works with the tools, prompts and answer checks of the
  repository below (it was trained with them).
- **Hardware:** one GPU with ≥ 40 GB of memory recommended (A100, H100, L40S; tested on an A100
  80 GB): the weights take 16 GB and a long stay needs a 32 000-token context.
- **Speed:** on one A100, about 1 200 stays per hour, measured on 200 stays of ~550 words
  processed together (7-8 tool calls per stay); the throughput is lower with fewer stays at a
  time. The encoder processes the same stays about 400 times faster.
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
suggested DP (`primary_diag_pred`), the DR (`dr`) and the model's `justification`.

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

Accuracy of the suggested **DP** against the PMSI DP of the stay, i.e. what `primary_diag`
contains: exact code, first 4 characters (the metric of the PARTAGES Codabench challenge), and
3-character category.

**Physician-reviewed AP-HP stays** (May 2026; text = acts + CRC + CRH + CR-OP), the same stays as for
the [encoder](https://huggingface.co/PARTAGES-CU2/cu2-icd10-dp-encoder). Views: **flattened** =
every review is a row; **agreement + single** = one row per stay, reviewed once or by several
reviewers who agree; **agreement** = only stays with several agreeing reviewers. **ICD-10** = all
stays; **surgery** = stays coded in the surgical scope of the use case (~3 500 codes).

| View | Scope | Stays | Exact | 4 char. | 3 char. |
|---|---|---:|---:|---:|---:|
| flattened | ICD-10 | 301 | 46.8 % | 48.5 % | 59.8 % |
| flattened | surgery | 253 | 51.8 % | 53.4 % | 64.0 % |
| agreement + single | ICD-10 | 190 | 50.0 % | 52.1 % | 63.2 % |
| agreement + single | surgery | 163 | 54.0 % | 55.8 % | 66.3 % |
| agreement | ICD-10 | 47 | 59.6 % | 59.6 % | 66.0 % |
| agreement | surgery | 42 | 64.3 % | 64.3 % | 66.7 % |

The agent is better than the encoder on these stays (flattened: 46.8 % vs 39.0 % exact), mostly
when the DP is a Z code (sessions, follow-up…): 47 % vs 24 % exact. Small sets: 95 % confidence
interval about ±7 points on ~200 stays, ±13 points on ~50; the reference coding is not certain
either. A few stays (about 3 %) get no answer within the limits of the agent and are not counted.

## Limitations

- Coding **assistance** only; the justification must be read critically.
- Slower than the encoder: several tool calls per stay.
- Mostly trained on surgical stays, with fictitious or synthetic records only.
- The codes used for training and evaluation are not certain, and some stays can legitimately be
  coded in more than one way.
- Evaluated on real stays of a single hospital group (AP-HP; AP-HP data used for evaluation only).
