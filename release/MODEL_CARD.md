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
departments, DIM).

**Terms used below**

- **DP** (*diagnostic principal*): the health problem that motivated the stay.
- **Z code**: an ICD-10 code of chapter XXI, which describes a reason for contact rather than a
  disease (e.g. `Z51.1` chemotherapy session, `Z09.0` follow-up after surgery).
- **DR** (*diagnostic relié*): when the DP is a Z code, the disease behind it (e.g. the cancer
  treated during a chemotherapy session).
- **CRC / CRH / CR-OP**: consultation reports, discharge summaries, operative reports.

**In short**

- **For:** coding assistance with an explanation the DIM can check (the justification and the
  searches made). **Not for:** automatic coding without validation; no associated diagnoses (DAS).
- **Runs locally** with [vLLM](https://github.com/vllm-project/vllm) and offline tools: no text
  leaves your infrastructure. Trained without any real patient data.
- **Needs its harness:** the model only works with the tools, prompts and answer checks of the
  repository below (it was trained with them).
- **Hardware:** one GPU with ≥ 40 GB of memory, or ≥ 24 GB with `--quantization fp8` (no
  measurable loss on the reviewed AP-HP stays, about 12 % slower).
- **Speed:** on one A100, about 1 200 stays per hour on 200 synthetic stays of ~550 words, and
  about 420 per hour on 339 real AP-HP stays (longer reports, ~11 tool calls per stay), all
  stays processed together; fewer stays at a time are slower.
- **Mostly trained on surgical stays**, but covers the whole ICD-10 (see Limitations).

**Everything to run and evaluate it on your stays:**
<https://github.com/aphp-datascience/partages-cu2-agent>.

## Quick use

```bash
git clone https://github.com/aphp-datascience/partages-cu2-agent.git && cd partages-cu2-agent && uv sync
uv run hf download PARTAGES-CU2/cu2-icd10-dp-agent --local-dir models/cu2-icd10-dp-agent
uv run python -m scripts.predict --model models/cu2-icd10-dp-agent --data my_stays.csv --out preds.csv
```

Output per stay: the suggested DP (`primary_diag_pred`), the DR (`dr`, only when the DP is a Z
code) and the model's `justification`; with `--keep-conv`, the whole conversation (searches, tool
results, answer).

## Input data

One CSV file, one row per hospital stay (RUM), with the columns of the PARTAGES guidelines (the
same format as the PARTAGES Codabench challenge). The repository turns it into the text the model
reads: the CCAM procedures on the first line (`Actes (CCAM) : CODE - label ; …`), then the reports.

| Column | Needed for | Content | Example |
|---|---|---|---|
| `id` | always | a unique id of the stay | `RUM_000123` |
| `speciality` | — | the specialty of the stay (not used by the model) | `CHIRURGIE VISCERALE` |
| `text` | always | the reports of the stay (CRC, CRH, CR-OP) concatenated | `Compte rendu de consultation …` |
| `procedure` | strongly recommended | the CCAM codes of the stay, separated by spaces | `NDPA014 NDPA008` |
| `primary_diag` | evaluation | the PMSI DP of the RUM, as coded (Z codes included) | `M6534`, `Z511` |

The repository's script compares the suggested DP with `primary_diag` (exact code, 4 and 3
characters).

## Model and training

### How the agent works

The model converses with a fixed harness (the repository's `agent/agent.py`). The *system
prompt* asks it to code the stay as a DIM physician would, following the ATIH MCO guide, defines
the DP and the DR, and ends with the mandatory coding procedure (`procedure.md` in this
repository); the *user message* is the record of the stay. At each turn the model either calls
tools (1 to 3 per message), whose results are appended to the conversation, or gives its final
answer as one line of JSON: `{"dp": "Z511", "dr": "C509", "justification": "..."}`.

Tool names, prompts and tool results are in French, as in training.

| Tool (offline, BM25) | Returns | Procedure step |
|---|---|---|
| `guide_mco` | the 3 closest passages of the ATIH MCO guide 2026 | which coding rule applies |
| `chercher_cim10` | up to 10 codes found by label, inclusion note or synonym, with their PMSI status | find candidate codes |
| `fiche_code` | label, PMSI status, inclusions, **exclusions**, parent and more precise codes | read the exclusions; check the final code |
| `comparer_codes` | all the codes of a category side by side | choose the 4th / 5th character |

An answer is refused, and the model asked again, if:

- a code does not exist, or is not allowed as DP / DR;
- one of the four tools was not called;
- the final code was not read with `fiche_code`.

After 12 turns, or when the next turn would not fit in the 32 000-token context,
the answer is requested without tools. Sampling: temperature 0.7, top-p 0.8, top-k 20, presence
penalty 1.0, Qwen3 thinking mode off; two runs on the same stays do not give exactly the same
codes.

### Training data

No real patient data was used for training. The model learns from about 27 600 conversations of
**Mistral Large 4** coding stays with the same tools, prompts and procedure:

| Stays coded | Conversations | Conversations kept |
|---|---:|---|
| [PARHAF](https://huggingface.co/datasets/HealthDataHub/PARHAF) (CC BY 4.0) fictitious patients and their rewrites | 6&nbsp;385 | all |
| PARHAF patients with a reference coding (Claude Opus, partly reviewed by physicians) | 784 | only conversations reaching the reference code |
| **MISTRAL** synthetic surgical stays generated by an LLM | 20&nbsp;428 | all (Mistral Large 4's coding replaces the noisy original code) |

**Why the conversations of a larger model?** Prompted with the same tools and procedure, the base
Qwen3-8B does not follow the PMSI coding well (it rarely codes a Z code as DP with its DR); it
learns it from the conversations of a larger model, Mistral Large 4. Only fictitious or synthetic
texts were sent to the API.

### Training parameters

- **Base model:** `Qwen/Qwen3-8B` (Apache-2.0).
- **Fine-tuning:** LoRA (rank 32, all projections), one epoch, learning rate 1e-4; the adapter is
  merged into the weights (bf16).
- **Examples:** one training example is one assistant turn of a conversation (tool call or final
  answer), with the conversation before it as context.

## Results (before any fine-tuning on real hospital data)

Accuracy of the suggested **DP** against the PMSI DP of the stay, i.e. what `primary_diag`
contains: exact code, first 4 characters (the metric of the PARTAGES Codabench challenge), and
3-character category.

**Physician-reviewed AP-HP stays** (May 2026, text = acts + CRC + CRH + CR-OP).

Views:

- **flattened**: every review is a row.
- **agreement + single**: one row per stay, reviewed once or by several reviewers who agree.
- **agreement**: only stays with several agreeing reviewers.

Scope: **ICD-10** = all stays, **surgery** = stays coded in the surgical scope of the use case
(~3 500 codes).

| View | Scope | Stays | Exact | 4 char. | 3 char. |
|---|---|---:|---:|---:|---:|
| flattened | ICD-10 | 310 | 45.5 % | 47.1 % | 58.1 % |
| flattened | surgery | 259 | 50.6 % | 52.1 % | 62.5 % |
| agreement + single | ICD-10 | 195 | 48.7 % | 50.8 % | 61.5 % |
| agreement + single | surgery | 167 | 52.7 % | 54.5 % | 64.7 % |
| agreement | ICD-10 | 47 | 59.6 % | 59.6 % | 66.0 % |
| agreement | surgery | 42 | 64.3 % | 64.3 % | 66.7 % |

- **No answer = error:** a stay without answer within the limits of the agent (9 of the 310
  flattened stays) counts as an error.
- **Z codes:** when the DP is a Z code (sessions, follow-up…), the agent finds it about as often as
  other DPs: 44.7 % exact (76 stays, flattened).
- **Small sets:** 95 % confidence interval about ±7 points on ~200 stays, ±13 points on ~50.
- **The reference coding is not certain either:** even reviewed PMSI coding contains errors, and
  some stays can legitimately be coded in more than one way. The **agreement** view, where
  reviewers concur, is the most reliable.

## Limitations

- **Mostly trained on surgery.** MISTRAL, by far the largest source, only covers ~4 000 codes,
  almost all surgical, and most PARHAF patients are surgical too. The model is clearly more
  accurate on surgical codes (compare the **surgery** rows above).
- Coding **assistance** only; the justification must be read critically.
- Slow: about ten tool calls per stay. About 3 % of the stays get no answer (mostly very long
  records that fill the context).
- One DP (and DR) per stay; no associated diagnoses.
- Trained on synthetic or fictitious notes: expect a domain shift on real hospital notes.
- **Reference codes are not certain.** The codes used for training and evaluation are not 100 %
  sure: synthetic and LLM-recoded labels contain errors, reviewed PMSI coding too, and some stays
  can legitimately be coded in more than one way.
- Evaluated on real stays of a single hospital group (AP-HP, single-RUM stays with CRC, CRH or
  CR-OP reports; AP-HP data used for evaluation only, never for training). Other report types
  and multi-RUM stays have not been evaluated specifically.
