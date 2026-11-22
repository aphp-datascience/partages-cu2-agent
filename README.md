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
(current version `v0`, a pre-release to test the agent; a final version will follow).

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
- **Hardware:** one GPU with ≥ 40 GB of memory recommended (A100, H100, L40S; tested on an A100
  80 GB): the weights take 16 GB and a long stay needs a 32 000-token context.

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

The same file as for the encoder: one **CSV file, one row per hospital stay (RUM)**, with the
columns of the PARTAGES guidelines (the format of the PARTAGES Codabench challenge):

| Column | Needed for | Content | Example |
|---|---|---|---|
| `id` | always | a unique id of the stay | `RUM_000123` |
| `speciality` | — | the specialty of the stay (not used by the model) | `CHIRURGIE VISCERALE` |
| `text` | always | the reports of the stay (CRC, CR-OP, CRH) concatenated, separated by a blank line or by the `[...]` marker of the guidelines | `Compte rendu de consultation …` |
| `procedure` | strongly recommended | the CCAM codes of the stay, separated by spaces; empty if none | `NDPA014 NDPA008` |
| `primary_diag` | evaluation | the PMSI principal diagnosis (DP) of the RUM, as coded, Z codes included | `M6534`, `Z511` |

Records longer than 60 000 characters are cut (the start is kept). For evaluation, build the test
set as carefully as for the encoder: stays reviewed by the DIM, ideally coded twice (see the
[encoder README](https://github.com/aphp-datascience/partages-cu2-encoder#2-input-data)).

## 3. Running and evaluating the agent

```bash
uv run python -m scripts.predict --model models/cu2-icd10-dp-agent --resources resources/data \
    --data my_test.csv --out preds.csv
```

| Option | What it does |
|---|---|
| `--model`, `--resources` | the downloaded model and resources folders |
| `--data`, `--out` | your stays file, and the file where predictions are written (CSV, or parquet if the name ends with `.parquet`) |
| `--no-acts` | do not write the CCAM line (by default it is built from `procedure`, as in training) |
| `--keep-conv` | also write the whole conversation (searches, tool results, answer) of each stay |
| `--procedure` | coding procedure of the system prompt; by default the `procedure.md` shipped with the model, i.e. the one it was trained with (keep it) |
| `--gpus` | number of GPUs (tensor parallelism) |

`preds.csv` has one row per stay:

| Column | Meaning |
|---|---|
| `id` | the id of the stay |
| `primary_diag_pred` | the suggested DP, to compare with your `primary_diag` |
| `dr` | the suggested DR (only when the DP is a Z code) |
| `target` | the disease of the stay: the DR if any, else the DP |
| `justification` | the model's explanation of its choice |
| `n_tools`, `forced_answer` | number of tool calls; whether the answer had to be requested (turn limit or very long record) |
| `conversation` | with `--keep-conv`: the whole conversation, JSON |

If the file has a `primary_diag` column, the script prints the share of stays where the suggested
DP equals it: exact code, first 4 characters (the Codabench metric) and first 3 characters.

## 4. Adapting the agent to your hospital

TODO (after the AP-HP results): self-distillation on your own coded stays. The agent plays each
training stay several times, the conversations that find your coding are kept, and the model is
fine-tuned on them (LoRA), without any larger model.

## Model and training

- **Base model:** [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) (Apache-2.0), fine-tuned
  with LoRA (rank 32) on agent conversations, the adapter being merged into the weights.
- **Training conversations** (version `v0`): about 15 000 tool-using conversations of a larger
  model, **Mistral Large 4**, coding stays with the same tools and procedure, no real patient data:
  4 903 on [PARHAF](https://huggingface.co/datasets/HealthDataHub/PARHAF) fictitious patients and
  their rewrites (kept when Mistral agrees with a second model), 10 428 on MISTRAL synthetic
  surgical stays. See the [model card](https://huggingface.co/PARTAGES-CU2/cu2-icd10-dp-agent).
- **Tools** (offline, BM25 search; no vector database): `chercher_cim10` (ICD-10 search over
  labels, inclusion notes and synonyms), `fiche_code` (code sheet: PMSI status, inclusions,
  exclusions), `comparer_codes` (all codes of a category side by side), `guide_mco` (passages of
  the ATIH MCO methodological guide 2026).
- **Procedure and checks:** the system prompt imposes the coding procedure
  ([`resources/procedure.md`](resources/procedure.md)); an answer is refused and the model asked
  again if a mandatory tool was not called, if a code does not exist or is not allowed as DP /
  DR, or if the final code was not checked with `fiche_code`. At most 12 tool turns, then the
  answer is requested.

## Results (before any fine-tuning on real hospital data)

TODO: physician-reviewed AP-HP stays.

Provisional, on fictitious patients: PARHAF CU2 test patients (288 patients written by physicians,
reviewed coding, never seen in training): 67.4 % exact on the target code (one run).

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
decoder/stays.py       reading the stays file (CSV of the guidelines)
scripts/predict.py     inference (and accuracy) on a file
resources/procedure.md coding procedure used when the model ships none (part of the system prompt)
data/dr_rule.json      keep-the-DR rule per Z code
data/ccam_2026.csv     CCAM 2026 nomenclature (ATIH)
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
