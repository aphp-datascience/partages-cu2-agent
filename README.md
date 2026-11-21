# PARTAGES-CU2: LLM agent for French ICD-10 medical coding

Given the reports of a French hospital stay, this agent codes the stay as in the PMSI (the French
hospital activity database): its **principal diagnosis (DP)** and, when the DP is a Z code, its
**related diagnosis (DR)**, in ICD-10 (CIM-10, ATIH version), with a **written justification**. A
fine-tuned 8-billion-parameter model (Qwen3-8B) reads the record, searches the ICD-10 and the ATIH
MCO methodological guide with offline tools, follows a mandatory coding procedure, and answers.
It was developed in the PARTAGES project (use case 2, coding assistance for medical information
departments, DIM), next to the
[encoder](https://github.com/aphp-datascience/partages-cu2-encoder), a lighter classifier.
The model is released on the Hugging Face Hub:
**[`PARTAGES-CU2/cu2-icd10-dp-agent`](https://huggingface.co/PARTAGES-CU2/cu2-icd10-dp-agent)**
(TODO: publication once the final version is chosen).

**Terms used below**

- **DP** (*diagnostic principal*): the health problem that motivated the stay.
- **Z code**: an ICD-10 code of chapter XXI, which describes a reason for contact rather than a
  disease (e.g. `Z51.1` chemotherapy session, `Z09.0` follow-up after surgery).
- **DR** (*diagnostic relié*): when the DP is a Z code, the disease behind it (e.g. the cancer
  treated during a chemotherapy session).
- **Target code**: the disease of the stay, i.e. the DR if there is one, otherwise the DP.
- **CRC / CRH / CR-OP**: consultation reports, discharge summaries, operative reports.

**In short**

- **For:** coding assistance with an explanation the DIM can check (the justification and the
  searches made). **Not for:** automatic coding without validation; no associated diagnoses (DAS).
- **Runs locally:** the model and the tools run on your machines (vLLM, offline indexes); no
  text leaves your infrastructure. The model was trained without any real patient data.
- **Encoder or agent?** The [encoder](https://github.com/aphp-datascience/partages-cu2-encoder)
  is faster and gives a ranked list of candidates; the agent is slower (several tool calls per
  stay) but explains its choice. TODO: compare their accuracy on AP-HP stays.
- **Hardware:** one GPU with ≥ 24 GB of memory (TODO: confirm; tested on A100 80 GB).

## 1. Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/); an NVIDIA GPU with a recent driver (vLLM).

```bash
git clone https://github.com/aphp-datascience/partages-cu2-decoder.git
cd partages-cu2-decoder
uv sync
```

**Model and resources.** The Hub repositories are private for now: ask the CU2 team for access,
log in, then download the model and the offline resources of the tools (ICD-10 index, MCO guide
passages, ICD-10 synonyms) once:

```bash
uv run huggingface-cli login
uv run huggingface-cli download PARTAGES-CU2/cu2-icd10-dp-agent --local-dir models/cu2-icd10-dp-agent
uv run huggingface-cli download PARTAGES-CU2/cu2-agent-resources --repo-type dataset \
    --local-dir resources/data
```

On a server without internet access, download on another machine and copy both folders over.

## 2. Input data

The same file as for the encoder: one parquet file, **one row per hospital stay (RUM)**.

| Column | Needed for | Content |
|---|---|---|
| `note_id` | always | a unique id of the stay |
| `note_text` | always | the CRC, CR-OP and CRH of the stay, in chronological order, separated by a blank line |
| `acts` | strongly recommended | the CCAM codes of the stay, e.g. `NFFC004;ZBQK002` (empty if none) |
| `dp` | evaluation | the PMSI principal diagnosis (`C509` or `C50.9`) |
| `dr` | evaluation | the PMSI related diagnosis (empty when none) |

Records longer than 60 000 characters are cut (the start is kept). For evaluation, build the test
set as carefully as for the encoder: stays reviewed by the DIM, ideally coded twice (see the
[encoder README](https://github.com/aphp-datascience/partages-cu2-encoder#2-input-data)).

## 3. Running and evaluating the agent

```bash
uv run python -m scripts.predict --model models/cu2-icd10-dp-agent --resources resources/data \
    --data my_test.parquet --out preds.parquet --acts acts --gold-dp dp --gold-dr dr
```

| Option | What it does |
|---|---|
| `--model`, `--resources` | the downloaded model and resources folders |
| `--data`, `--out` | your parquet file, and the file where predictions are written |
| `--acts acts` | name of the column holding the CCAM codes: the script writes the `Actes (CCAM) : …` line at the top of each text, as in training. Leave it out if you have no acts |
| `--gold-dp dp`, `--gold-dr dr` | names of the columns holding **your** PMSI coding. Optional: only used to print the accuracy at the end |
| `--keep-conv` | also write the whole conversation (searches, tool results, answer) of each stay |
| `--gpus` | number of GPUs (tensor parallelism) |

`preds.parquet` has one row per stay:

| Column | Meaning |
|---|---|
| `dp`, `dr` | the suggested PMSI coding (DR only when the DP is a Z code) |
| `target` | the target code, to compare with your coding: the DR if any, else the DP |
| `justification` | the model's explanation of its choice |
| `n_tools`, `forced_answer` | number of tool calls; whether the answer had to be requested (turn limit or very long record) |
| `conversation` | with `--keep-conv`: the whole conversation, JSON |

## 4. Adapting the agent to your hospital

TODO (after the AP-HP results): self-distillation on your own coded stays. The agent plays each
training stay several times, the conversations that find your coding are kept, and the model is
fine-tuned on them (LoRA), without any larger model.

## Model and training

TODO: final version (E / F) and figures.

- **Base model:** [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) (Apache-2.0), fine-tuned
  with LoRA (rank 32) on agent conversations, the adapter being merged into the weights.
- **Training conversations:** tool-using conversations of a larger model, **Mistral Large 4**,
  coding synthetic and fictitious stays (no real patient data) with the same tools and procedure:
  MISTRAL synthetic surgical stays, PARHAF fictitious patients and their rewrites. Only
  conversations ending on an accepted code were kept. TODO: volumes, Claude Opus conversations.
- **Tools** (offline, BM25 search; no vector database): `chercher_cim10` (ICD-10 search over
  labels, inclusion notes and synonyms), `fiche_code` (code sheet: PMSI status, inclusions,
  exclusions), `comparer_codes` (all codes of a category side by side), `guide_mco` (passages of
  the ATIH MCO methodological guide 2026).
- **Procedure and checks:** the system prompt imposes the coding procedure
  ([`resources/procedure.md`](resources/procedure.md)); an answer is refused and the model asked
  again if a mandatory tool was not called, if a code does not exist or is not allowed as DP /
  DR, or if the final code was not checked with `fiche_code`. At most 12 tool turns, then the
  answer is requested.

## Results

TODO: AP-HP reviewed stays and test set (before any fine-tuning on real hospital data).

## Limitations

- Coding **assistance** only; the justification must be read critically.
- Slower than the encoder: several tool calls per stay.
- Mostly trained on surgical stays (as the encoder), with fictitious or synthetic records only.
- The codes used for training and evaluation are not certain, and some stays can legitimately be
  coded in more than one way.
- Evaluated on real stays of a single hospital group (AP-HP).

## Repository layout

```
decoder/agent.py       the agent: prompts, conversation loop, answer checks
decoder/tools.py       the four offline tools
decoder/acts.py        CCAM procedure line of the input text
scripts/predict.py     inference (and accuracy) on a file
resources/procedure.md mandatory coding procedure (part of the system prompt)
data/dr_rule.json      keep-the-DR rule per Z code
data/ccam_2026.csv     CCAM 2026 nomenclature (ATIH)
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
