"""Run the coding agent on a parquet file of stays (one row per stay / RUM).

    uv run python -m scripts.predict --model models/cu2-icd10-dp-agent --resources resources/data \
        --data my_test.parquet --out preds.parquet --acts acts [--gold-dp dp --gold-dr dr]

Input columns: `note_id`, `note_text` (the reports of the stay, chronological order), optionally a column
of CCAM codes (`--acts`) and your PMSI coding (`--gold-dp`, `--gold-dr`, only used to print the accuracy).
Output: one row per stay with the suggested `dp`, `dr`, the `target` (DR if any, else DP), the
`justification` written by the model, the number of tool calls and, with `--keep-conv`, the whole
conversation (to read how the model reached its answer).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decoder.acts import load_ccam, with_acts  # noqa: E402
from decoder.agent import Settings, run, target  # noqa: E402
from decoder.tools import Ressources, norm_code  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="local dir or Hugging Face repo id of the model")
    ap.add_argument("--resources", type=Path, default=ROOT / "resources/data",
                    help="folder of the downloaded resources (index/, syn_all.parquet)")
    ap.add_argument("--data", required=True, type=Path, help="parquet file or folder of parquet files")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--acts", help="column of CCAM codes: the 'Actes (CCAM)' line is added to the text")
    ap.add_argument("--gold-dp", help="column of your PMSI DP (optional, accuracy only)")
    ap.add_argument("--gold-dr", help="column of your PMSI DR (optional, accuracy only)")
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

    df = pd.read_parquet(a.data)
    texts = df["note_text"].astype(str)
    if a.acts:
        ccam = load_ccam()
        texts = [with_acts(t, c, ccam) for t, c in zip(texts, df[a.acts])]
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
        "note_id": df["note_id"].to_numpy(),
        "dp": [r["dp"] for r in res],
        "dr": [r["dr"] for r in res],
        "target": [target(r["dp"], r["dr"], rule) for r in res],
        "justification": [r["justification"] for r in res],
        "n_tools": [r["n_tools"] for r in res],
        "forced_answer": [r["forced"] for r in res],
    })
    if a.keep_conv:
        out["conversation"] = [r["conv"] for r in res]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"-> {a.out} ({time.time() - t0:.0f} s) ; no answer {out['dp'].isna().mean():.1%} ; "
          f"forced answer {out['forced_answer'].mean():.1%} ; tools / stay {out['n_tools'].mean():.1f}")

    if a.gold_dp:
        def code(x):
            return norm_code(x) if isinstance(x, str) and x.strip() else None

        gdp = df[a.gold_dp].map(code)
        gdr = df[a.gold_dr].map(code) if a.gold_dr else pd.Series([None] * len(df))
        gold = [target(p, r) for p, r in zip(gdp, gdr)]  # your coding as is: DR if any, else DP
        pred = [t if isinstance(t, str) else None for t in out["target"]]
        ok = pd.Series([t is not None and t == g for t, g in zip(pred, gold)])
        ok3 = pd.Series([t is not None and g is not None and t[:3] == g[:3] for t, g in zip(pred, gold)])
        print(f"target vs your coding: exact {ok.mean():.1%}, 3 characters {ok3.mean():.1%} (n={len(ok)})")


if __name__ == "__main__":
    main()
