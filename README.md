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
(current version `v0.1`, a pre-release to test the agent; a final version will follow).

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
- **Encoder or agent?** The [encoder](https://github.com/aphp-datascience/partages-cu2-encoder)
  is faster and gives a ranked list of candidates; the agent is slower (several tool calls per
  stay) but explains its choice. On physician-reviewed AP-HP stays the agent finds the exact DP
  more often (46.8 % vs 41.0 %), mostly for Z codes (sessions, follow-up); the encoder's top-20
  contains the DP in 71 % of the stays, and only the encoder can be fine-tuned on your stays.
- **Hardware:** one GPU with ≥ 40 GB of memory recommended (A100, H100, L40S; tested on an A100
  80 GB): the weights take 16 GB and a long stay needs a 32 000-token context.
- **Speed:** on one A100, about 1 200 stays per hour, measured on 200 stays of ~550 words
  processed together (7-8 tool calls per stay); the throughput is lower with fewer stays at a
  time. The encoder processes the same stays about 400 times faster.

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

## 4. Adapting the agent to your hospital

No adaptation of the agent is provided yet. We tested **self-distillation** on AP-HP coded stays
(the agent plays each training stay several times, the conversations that find the hospital's
coding are kept, and the model is fine-tuned on them): it brought no gain (−0.8 points, not
significant), because the model only learns again the stays it already solves. Adapting with
conversations of a larger model is being studied. To adapt the coding to your hospital today,
fine-tune the [encoder](https://github.com/aphp-datascience/partages-cu2-encoder#4-fine-tuning-on-your-stays).

## How the agent works

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

**The tools** (offline; each takes one text argument and returns French text to the model):

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

## Model and training

- **Base model:** [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) (Apache-2.0), fine-tuned
  with LoRA (rank 32) on agent conversations, the adapter being merged into the weights.
- **Training conversations** (version `v0.1`): about 27 600 tool-using conversations of a larger
  model, **Mistral Large 4**, coding stays with the same tools and procedure, no real patient data:
  6 385 on [PARHAF](https://huggingface.co/datasets/HealthDataHub/PARHAF) fictitious patients and
  their rewrites, 784 on PARHAF patients with a reference coding (only conversations reaching it),
  20 428 on MISTRAL synthetic surgical stays. See the
  [model card](https://huggingface.co/PARTAGES-CU2/cu2-icd10-dp-agent).
- **Tools** (offline, BM25 search; no vector database): `chercher_cim10` (ICD-10 search over
  labels, inclusion notes and synonyms), `fiche_code` (code sheet: PMSI status, inclusions,
  exclusions), `comparer_codes` (all codes of a category side by side), `guide_mco` (passages of
  the ATIH MCO methodological guide 2026).
- **Procedure and checks:** the system prompt imposes the coding procedure
  ([`resources/procedure.md`](resources/procedure.md)); an answer is refused and the model asked
  again if a mandatory tool was not called, if a code does not exist or is not allowed as DP /
  DR, or if the final code was not checked with `fiche_code`. At most 12 tool turns, then the
  answer is requested. Details in [How the agent works](#how-the-agent-works).

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

The agent is better than the encoder on these stays (flattened: 46.8 % vs 41.0 % exact), mostly
when the DP is a Z code (sessions, follow-up…): 47 % vs 33 % exact. Small sets: 95 % confidence
interval about ±7 points on ~200 stays, ±13 points on ~50; the reference coding is not certain
either. A few stays (about 3 %) get no answer within the limits of the agent and are not counted.

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
data/ccam_2026.csv     CCAM 2026 nomenclature (ATIH)
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
