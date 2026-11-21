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
- **Hardware:** one GPU with ≥ 24 GB of memory (TODO: confirm).

**Everything to run and evaluate it on your stays:**
<https://github.com/aphp-datascience/partages-cu2-decoder>.

## Quick use

```bash
git clone https://github.com/aphp-datascience/partages-cu2-decoder.git && cd partages-cu2-decoder && uv sync
uv run huggingface-cli download PARTAGES-CU2/cu2-icd10-dp-agent --local-dir models/cu2-icd10-dp-agent
uv run huggingface-cli download PARTAGES-CU2/cu2-agent-resources --repo-type dataset --local-dir resources/data
uv run python -m scripts.predict --model models/cu2-icd10-dp-agent --resources resources/data \
    --data my_stays.parquet --out preds.parquet --acts acts
```

Input: one row per stay with `note_id`, `note_text` (consultation, operative and discharge reports
in chronological order) and, if available, the CCAM codes of the stay (`acts`). Output: `dp`,
`dr`, `target`, `justification` per stay.

## Model and training

TODO: final version and figures.

- **Base model:** `Qwen/Qwen3-8B` (Apache-2.0); LoRA fine-tuning (rank 32), adapter merged into
  the weights (bf16).
- **Training data:** tool-using conversations of **Mistral Large 4** (and TODO: Claude Opus) coding
  synthetic surgical stays (MISTRAL) and fictitious patients (PARHAF, CC BY 4.0) with the same tools
  and procedure; only conversations ending on an accepted code were kept. No real patient data.
- **Tools:** ICD-10 search (labels, inclusion notes, synonyms; BM25), code sheet (PMSI status,
  inclusions, exclusions), comparison of the codes of a category, search in the ATIH MCO guide 2026.

## Results (before any fine-tuning on real hospital data)

TODO: physician-reviewed AP-HP stays.

## Limitations

- Coding **assistance** only; the justification must be read critically.
- Slower than the encoder: several tool calls per stay.
- Mostly trained on surgical stays, with fictitious or synthetic records only.
- The codes used for training and evaluation are not certain, and some stays can legitimately be
  coded in more than one way.
- Evaluated on real stays of a single hospital group (AP-HP; AP-HP data used for evaluation only).
