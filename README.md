# PARTAGES-CU2: LLM agent for French ICD-10 medical coding

Given the reports of a French hospital stay, this agent codes the stay as in the PMSI (the French
hospital activity database): its **principal diagnosis (DP)** and, when the DP is a Z code, its
**related diagnosis (DR)**, in ICD-10 (CIM-10, ATIH version), with a **written justification**. A
fine-tuned 8-billion-parameter model (Qwen3-8B) reads the record, searches the ICD-10 and the ATIH
MCO methodological guide with offline tools, follows a mandatory coding procedure, and answers.
It was developed in the PARTAGES project (use case 2, coding assistance for medical information
departments, DIM). The model is released on the Hugging Face Hub:
**[`PARTAGES-CU2/cu2-icd10-dp-agent`](https://huggingface.co/PARTAGES-CU2/cu2-icd10-dp-agent)**.

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
- **Runs locally:** the model and the tools run on your machines (vLLM, offline indexes); no
  text leaves your infrastructure. The model was trained without any real patient data.
- **Hardware:** one GPU with ≥ 40 GB of memory, or ≥ 24 GB with `--quantization fp8` (no
  measurable loss on the reviewed AP-HP stays, about 12 % slower).
- **Speed:** on one A100, about 1 200 stays per hour on 200 synthetic stays of ~550 words, and
  about 420 per hour on 339 real AP-HP stays (longer reports, ~11 tool calls per stay), all
  stays processed together; fewer stays at a time are slower.
- **Mostly trained on surgical stays**, but covers the whole ICD-10 (see
  [Limitations](#limitations)).

## 1. Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/); an NVIDIA GPU with a recent driver (vLLM).

```bash
git clone https://github.com/aphp-datascience/partages-cu2-decoder.git
cd partages-cu2-decoder
uv sync
```

**Model.** The Hub repository is private for now: ask the CU2 team for access, log in, then
download the model once. It ships the offline resources of the tools too (ICD-10 index, MCO guide
passages, ICD-10 synonyms):

```bash
uv run hf auth login
uv run hf download PARTAGES-CU2/cu2-icd10-dp-agent --local-dir models/cu2-icd10-dp-agent
```

On a server without internet access, download on another machine and copy the model folder over.

## 2. Input data

One **CSV file, one row per hospital stay (RUM)**, with the columns of the PARTAGES guidelines
(the format of the PARTAGES Codabench challenge):

| Column | Needed for | Content | Example |
|---|---|---|---|
| `id` | always | a unique id of the stay | `RUM_000123` |
| `speciality` | — | the specialty of the stay (not used by the model) | `CHIRURGIE VISCERALE` |
| `text` | always | the reports of the stay (CRC, CRH, CR-OP) concatenated, separated by a blank line or by the `[...]` marker of the guidelines | `Compte rendu de consultation …` |
| `procedure` | strongly recommended | the CCAM codes of the stay, separated by spaces; empty if none | `NDPA014 NDPA008` |
| `primary_diag` | evaluation | the PMSI principal diagnosis (DP) of the RUM, as coded, Z codes included | `M6534`, `Z511` |

Records longer than 60 000 characters are cut (the start is kept). **Make the test set as clean as
possible:** a noisy test set scores the model against coding errors. If you can, use stays reviewed
by the DIM, ideally coded twice independently, and keep only the stays where both codings agree. A
raw accuracy must not be read as a ranking of hospitals, whose coding practices differ.

## 3. Running and evaluating the agent

```bash
uv run python -m scripts.predict --model models/cu2-icd10-dp-agent --data my_test.csv --out preds.csv
```

| Option | What it does |
|---|---|
| `--model` | the downloaded model folder (its `resources/` folder holds the tool resources) |
| `--data`, `--out` | your stays file, and the file where predictions are written (CSV, or parquet if the name ends with `.parquet`) |
| `--no-acts` | do not write the CCAM line (by default it is built from `procedure`, as in training) |
| `--keep-conv` | also write the whole conversation (searches, tool results, answer) of each stay |
| `--procedure` | coding procedure of the system prompt; by default the `procedure.md` shipped with the model, i.e. the one it was trained with (keep it) |
| `--gpus` | number of GPUs (tensor parallelism) |
| `--quantization fp8` | load the weights in FP8, for a GPU with 24 GB of memory (see Hardware above) |
| `--max-turns` | tool turns before the answer is requested (default 12, as in training) |
| `--max-model-len` | context length in tokens (default 32 768) |
| `--max-chars` | characters of the record kept, from the start (default 60 000) |

`preds.csv` has one row per stay:

| Column | Meaning |
|---|---|
| `id` | the id of the stay |
| `primary_diag_pred` | the suggested DP, to compare with your `primary_diag` |
| `dr` | the suggested DR (only when the DP is a Z code) |
| `justification` | the model's explanation of its choice |
| `n_tools`, `forced_answer` | number of tool calls; whether the answer had to be requested (turn limit or very long record) |
| `conversation` | with `--keep-conv`: the whole conversation, JSON |

If the file has a `primary_diag` column, the script prints the share of stays where the suggested
DP equals it: exact code, first 4 characters (the Codabench metric) and first 3 characters.

## Model and training

### How the agent works

The agent is a conversation between the model and a fixed harness (`decoder/agent.py`). The model was
trained on conversations produced by exactly this harness, so **keep the prompts, the procedure and the
checks as they are**: change them only if you fine-tune the model again.

**The prompt.** The *system prompt* asks the model to code the stay as a DIM physician would, following
the ATIH MCO guide. It defines the DP (a disease, a symptom, or a Z code of care: sessions, negative
follow-up Z08/Z09, device care Z43-Z47, palliative care…) and the DR (only when the DP is a Z code: the
disease behind that care), lists the tools, and ends with the mandatory coding procedure
[`resources/procedure.md`](resources/procedure.md) (the copy shipped with the model is used if present).
The *user message* is the record of the stay (CCAM line, then the reports) followed by « Quel est le
diagnostic principal ? ». The final answer is one line of JSON:
`{"dp": "Z511", "dr": "C509", "justification": "..."}` (codes without dot, `dr` null if none).

**The tools** (offline; each takes one text argument and returns text to the model). Tool names,
prompts and tool results are in French, as in training:

| Tool | Argument | Returns | Procedure step |
|---|---|---|---|
| `guide_mco` | a question on the clinical situation | the 3 closest passages of the MCO guide 2026 (BM25) | b. which coding rule applies |
| `chercher_cim10` | a medical term | up to 10 codes found by label, inclusion note or synonym, each with its PMSI status (allowed as DP or not) | c. find candidate codes |
| `fiche_code` | a code or a category | label, PMSI status, inclusions, **exclusions**, coding notes, parent and more precise codes | d. read the exclusions; f. check the final code |
| `comparer_codes` | a category (3 characters) | all the leaf codes of the category side by side, with their inclusions and exclusions | e. choose the 4th / 5th character |

**A turn.** At each turn the model writes one message:

```
system: role, DP / DR definitions, tools, procedure      user: record of the stay
  │
  ├─ turn 1..12: assistant message
  │     ├─ with tool calls (1 to 3) ──► the harness runs them, appends the results ──► next turn
  │     └─ without tool call = final answer (JSON) ──► checks
  │                                                     ├─ OK ──► done
  │                                                     └─ refused ──► correction request ──► next turn
  └─ after 12 turns, or when the next turn would not fit in the context:
        « Plus d'appel d'outil : donne maintenant ta réponse JSON. » ──► one last answer, no tools
```

Stays are processed together in batches (vLLM): every unfinished conversation advances by one turn at a time.

**The checks** on an answer, in this order, each refusal being sent back to the model as a user message:

| Check | If it fails |
|---|---|
| the answer is the JSON object | asked again for the JSON |
| the DP and the DR exist in the ICD-10 (ATIH) | asked for an existing leaf code |
| the DP is allowed as DP, the DR as DR (PMSI status) | asked for a more precise or another code |
| the 4 tools of the procedure were all called | reminded to call the missing ones (steps b to f), up to 4 times |
| the final code (DR, else DP) was read with `fiche_code` | asked to check it and compare it with the other candidates |

The procedure reminder can be sent 4 times; the other refusals share a budget of 2 retries per stay.
When the retries are used up, the last answer is kept as it is (if its DP is a valid code). A DR given with
a DP that is not a Z code is dropped without asking. A message cut at 3 072 tokens is generated again
(this uses the same budget of 2 retries); after that, or when even the final answer no longer fits in
the context, the stay gets no answer (about 3 % of the AP-HP reviewed stays).

**Generation settings.** Temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.0, Qwen3 *thinking*
mode off (the model writes its messages and tool calls directly, as in its training conversations). The
sampling is random: **two runs on the same file do not give exactly the same codes**; compare models on
the same stays and expect a few points of variation on small test sets.

To see all this on a stay, run `scripts.predict` with `--keep-conv` and read the `conversation` column.

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

**Physician-reviewed AP-HP stays** (May 2026; text = acts + CRC + CRH + CR-OP). Views: **flattened** =
every review is a row; **agreement + single** = one row per stay, reviewed once or by several
reviewers who agree; **agreement** = only stays with several agreeing reviewers. **ICD-10** = all
stays; **surgery** = stays coded in the surgical scope of the use case (~3 500 codes).

| View | Scope | Stays | Exact | 4 char. | 3 char. |
|---|---|---:|---:|---:|---:|
| flattened | ICD-10 | 310 | 45.5 % | 47.1 % | 58.1 % |
| flattened | surgery | 259 | 50.6 % | 52.1 % | 62.5 % |
| agreement + single | ICD-10 | 195 | 48.7 % | 50.8 % | 61.5 % |
| agreement + single | surgery | 167 | 52.7 % | 54.5 % | 64.7 % |
| agreement | ICD-10 | 47 | 59.6 % | 59.6 % | 66.0 % |
| agreement | surgery | 42 | 64.3 % | 64.3 % | 66.7 % |

A stay without answer within the limits of the agent (9 of the 310 flattened stays) counts as an
error. When the DP is a Z code (sessions, follow-up…), the agent finds it about as often as other
DPs: 44.7 % exact (76 stays, flattened). Small sets: 95 % confidence interval about ±7 points on
~200 stays, ±13 points on ~50. The reference coding is not certain either.

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

## Repository layout

```
decoder/agent.py       the agent: prompts, conversation loop, answer checks
decoder/tools.py       the four offline tools
decoder/acts.py        CCAM procedure line of the input text
decoder/stays.py       reading the stays file (CSV of the guidelines)
scripts/predict.py     inference (and accuracy) on a file
resources/procedure.md coding procedure used when the model ships none (part of the system prompt)
data/ccam_2026.csv     CCAM 2026 nomenclature (ATIH)
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
