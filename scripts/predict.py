"""Run the coding agent on a stays file (one row per stay / RUM).

    uv run python -m scripts.predict --model models/cu2-icd10-dp-agent --resources resources/data \
        --data my_test.csv --out preds.csv

Input: the stays file of the PARTAGES guidelines (decoder/stays.py): CSV with `id`, `speciality`, `text`,
`procedure`, `primary_diag` (the last one optional, only used to print the accuracy). The CCAM codes of
`procedure` are written as the first line of the text, as in training (`--no-acts` to skip).
Output (CSV, or parquet if --out ends with .parquet): one row per stay with the suggested DP
(`primary_diag_pred`), `dr`, the `target` (DR if any, else DP), the model's `justification`, the number of
tool calls and, with `--keep-conv`, the whole conversation (to read how the model reached its answer).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decoder.acts import load_ccam, with_acts  # noqa: E402
from decoder.stays import read_stays  # noqa: E402
from decoder.agent import Settings, run, target  # noqa: E402
from decoder.tools import Ressources  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="local dir or Hugging Face repo id of the model")
    ap.add_argument("--resources", type=Path, default=ROOT / "resources/data",
                    help="folder of the downloaded resources (index/, syn_all.parquet)")
    ap.add_argument("--data", required=True, type=Path, help="stays file (CSV or parquet)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-acts", action="store_true", help="do not add the CCAM line")
    ap.add_argument("--rule", type=Path, default=ROOT / "data/dr_rule.json",
                    help="keep-the-DR rule per Z code for `target`; 'none' keeps every DR")
    ap.add_argument("--max-chars", type=int, default=60000, help="characters of the record kept (start)")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpus", type=int, default=1, help="tensor parallel GPUs")
    ap.add_argument("--procedure", type=Path,
                    help="coding procedure of the system prompt; default: procedure.md shipped with the model (the "
                         "one it was trained with), else resources/procedure.md")
    ap.add_argument("--keep-conv", action="store_true", help="also write the whole conversation")
    a = ap.parse_args()

    df = read_stays(a.data)
    texts = list(df["text"])
    if not a.no_acts:
        ccam = load_ccam()
        texts = [with_acts(t, c, ccam) for t, c in zip(texts, df["procedure"])]
    long = sum(len(t) > a.max_chars for t in texts)
    texts = [t[: a.max_chars] for t in texts]
    print(f"{len(texts)} stays ; {long} cut to {a.max_chars} characters", flush=True)

    model = Path(a.model)
    if not model.is_dir():  # Hugging Face repo id ("org/name" or "org/name@revision")
        from huggingface_hub import snapshot_download

        repo, _, rev = a.model.partition("@")
        model = Path(snapshot_download(repo, revision=rev or None))
    procedure = a.procedure or (model / "procedure.md" if (model / "procedure.md").exists() else ROOT / "resources/procedure.md")
    print(f"model {model} ; procedure {procedure}", flush=True)
    R = Ressources(a.resources / "index", a.resources / "syn_all.parquet")
    from vllm import LLM

    # cascade attention off: with small batches sharing the long system prompt, vLLM 0.10 produced degenerate
    # outputs (repetitions, no answer for ~60 % of 30 stays; 0 % without it)
    llm = LLM(model=str(model), max_model_len=a.max_model_len, gpu_memory_utilization=0.9,
              tensor_parallel_size=a.gpus, enable_prefix_caching=True, seed=0, disable_cascade_attn=True)
    t0 = time.time()
    res = run(llm, texts, R, Settings(max_turns=a.max_turns, max_model_len=a.max_model_len, procedure=procedure))
    rule = None if str(a.rule) == "none" else json.loads(a.rule.read_text())
    out = pd.DataFrame({
        "id": df["id"].to_numpy(),
        "primary_diag_pred": [r["dp"] for r in res],
        "dr": [r["dr"] for r in res],
        "target": [target(r["dp"], r["dr"], rule) for r in res],
        "justification": [r["justification"] for r in res],
        "n_tools": [r["n_tools"] for r in res],
        "forced_answer": [r["forced"] for r in res],
    })
    if a.keep_conv:
        out["conversation"] = [r["conv"] for r in res]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.suffix.lower() == ".parquet":
        out.to_parquet(a.out, index=False)
    else:
        out.to_csv(a.out, index=False)
    print(f"-> {a.out} ({time.time() - t0:.0f} s) ; no answer {out['primary_diag_pred'].isna().mean():.1%} ; "
          f"forced answer {out['forced_answer'].mean():.1%} ; tools / stay {out['n_tools'].mean():.1f}")

    if "primary_diag" in df:
        g = df["primary_diag"]
        pred = pd.Series([p if isinstance(p, str) else "" for p in out["primary_diag_pred"]])
        s_ = g.notna().to_numpy()
        gg, pp = g[s_].reset_index(drop=True), pred[s_].reset_index(drop=True)
        print(f"suggested DP vs primary_diag, n={int(s_.sum())}: exact {(pp == gg).mean():.1%}, "
              f"4 characters {(pp.str[:4] == gg.str[:4]).mean():.1%}, 3 characters {(pp.str[:3] == gg.str[:3]).mean():.1%}")

if __name__ == "__main__":
    main()
