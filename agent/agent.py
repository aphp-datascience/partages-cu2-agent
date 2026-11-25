"""PMSI coding agent: a Qwen3-8B fine-tuned model reads the record of a stay, calls offline tools (ICD-10
search, code sheet, comparison of the codes of a category, ATIH MCO guide) following a mandatory coding
procedure, and answers the DP and the DR of the stay with a justification.

Conversations advance in lockstep batches with vLLM: at each turn every unfinished conversation generates
one assistant message; tool calls are executed and their results appended as `tool` messages; a message
without tool call must hold the final answer {"dp": ..., "dr": ... | null, "justification": ...}. Answers
are checked (existing ICD-10 code, allowed as DP / DR, mandatory tools called, final code checked with
fiche_code) and the model is asked to correct them otherwise. After `max_turns`, or when the next turn would
not fit in the context, the model is asked for its final answer without tools.

The system prompt, the user message and the checks are those the released model was trained with: do not
change them unless you fine-tune the model again.
"""

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .tools import TOOLS, Ressources, norm_code

TOOL_NAMES = ["chercher_cim10", "fiche_code", "guide_mco", "comparer_codes"]
REQUIRED = ["guide_mco", "chercher_cim10", "fiche_code", "comparer_codes"]
PROCEDURE = Path(__file__).resolve().parent.parent / "resources" / "procedure.md"

TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
XML_FN = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
XML_PARAM = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
THINK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
CODE_OK = re.compile(r"^[A-Z][0-9]{2}[0-9A-Z]{0,4}(\+[0-9]{1,2})?$")
ANSWER = re.compile(r"\{[^{}]*\"dp\"\s*:\s*\"([^\"]+)\"[^{}]*\}", re.DOTALL)
DR_FIELD = re.compile(r"\"dr\"\s*:\s*(?:\"([^\"]*)\"|null)")
JUSTIF = re.compile(r"\"justification\"\s*:\s*\"((?:[^\"\\]|\\.)*)\"", re.DOTALL)

SYSTEM = """Tu es un médecin DIM expert du codage PMSI MCO en France. On te donne le dossier d'un séjour \
(comptes rendus). Ta tâche : coder le séjour comme dans le RUM, selon le Guide méthodologique MCO de l'ATIH.
- Le DIAGNOSTIC PRINCIPAL (DP) est le problème de santé qui a motivé l'admission, déterminé à la sortie. \
Selon la situation clinique, c'est une maladie, un symptôme, ou un code Z de prise en charge (séances de \
chimiothérapie, de radiothérapie, de dialyse ; surveillance négative Z08/Z09 ; soins de stomie, ablation ou \
ajustement de matériel Z43-Z47 ; soins palliatifs Z51.5…).
- Le DIAGNOSTIC RELIÉ (DR) n'existe que si le DP est un code Z : c'est la maladie chronique ou de longue durée, \
ou l'état permanent, en rapport avec cette prise en charge (ex. le cancer traité par la chimiothérapie). \
Pas de DR si le DP n'est pas un code Z, ni si aucune maladie de ce type ne justifie la prise en charge.
Chaque code doit être le plus précis possible (une feuille de la CIM-10).
{outils}
Quand tu es sûr, réponds UNIQUEMENT par un objet JSON sur une ligne, sans autre texte :
{{"dp": "XXXX", "dr": "YYYY" ou null, "justification": "..."}} où les codes CIM-10 sont sans point."""

TOOL_TXT = {"chercher_cim10": "chercher dans la CIM-10 (chercher_cim10)",
            "fiche_code": "lire la fiche d'un code (fiche_code)",
            "guide_mco": "consulter le Guide méthodologique MCO (guide_mco)",
            "comparer_codes": "comparer les codes d'une même catégorie pour choisir le code précis (comparer_codes)"}
FORCE = "Plus d'appel d'outil : donne maintenant ta réponse JSON."


def system_prompt(max_turns: int, procedure: Path = PROCEDURE) -> str:
    outils = (f"Tu disposes d'outils pour {', '.join(TOOL_TXT[x] for x in TOOL_NAMES)}. Vérifie le code choisi "
              f"avec fiche_code avant de répondre. Fais au plus {max_turns} appels d'outils.")
    return SYSTEM.format(outils=outils) + "\n\n" + procedure.read_text(encoding="utf-8").strip()


def user_message(text: str) -> str:
    return f"Dossier du séjour :\n\"\"\"\n{text.strip()}\n\"\"\"\n\nQuel est le diagnostic principal ?"


def parse_call(inner: str) -> tuple[str | None, dict]:
    """One tool call, JSON (Qwen3) or XML <function=..><parameter=..> format."""
    inner = inner.strip()
    m = XML_FN.search(inner)
    if m:
        return m.group(1), {k: v for k, v in XML_PARAM.findall(m.group(2))}
    try:
        d = json.loads(inner)
        a = d.get("arguments") or {}
        return d.get("name"), (json.loads(a) if isinstance(a, str) else a)
    except (json.JSONDecodeError, AttributeError):
        return None, {}


def parse_answer(text: str) -> tuple[str | None, str | None, str | None]:
    """(DP, DR, justification); DR None if absent, null or empty."""
    m = ANSWER.search(THINK.sub("", text))
    if not m:
        return None, None, None
    d = DR_FIELD.search(m.group(0))
    dr = d.group(1) if d and d.group(1) else None
    j = JUSTIF.search(m.group(0))
    why = json.loads(f'"{j.group(1)}"') if j else None
    return norm_code(m.group(1)), (norm_code(dr) if dr else None), why


@dataclass
class Settings:
    max_turns: int = 12
    max_tokens: int = 3072
    max_model_len: int = 32768
    temperature: float = 0.7
    top_p: float = 0.8
    presence_penalty: float = 1.0
    procedure: Path = PROCEDURE  # must be the procedure the model was trained with


def check(R: Ressources, dp: str | None, dr: str | None) -> tuple[str | None, str | None]:
    """(reason to refuse the answer, or None ; DR kept). Same order of checks as in training."""
    if dp is None:
        return ("Réponds uniquement par l'objet JSON demandé : {\"dp\": \"...\", \"dr\": \"...\" ou null, "
                "\"justification\": \"...\"}."), dr
    if not CODE_OK.match(dp) or R.cim.get(dp) is None:
        return f"« {dp} » n'est pas un code CIM-10 FR valide. Donne un code existant (feuille).", dr
    if dr and not dp.startswith("Z"):
        return None, None  # no DR when the DP is not a Z code: dropped silently
    if dr and (not CODE_OK.match(dr) or R.cim.get(dr) is None):
        return f"DR « {dr} » n'est pas un code CIM-10 FR valide. Donne un code existant (feuille) ou null.", dr
    if R.cim[dp].get("mco_had") not in (None, "", "0"):
        return (f"{dp} ({R.libelle(dp)}) ne peut pas être DP : {R.statut_dp(dp)}. "
                "Choisis un code plus précis (feuille) ou un autre code autorisé en DP."), dr
    if dr and R.cim[dr].get("mco_had") in ("1", "2", "3"):
        return (f"{dr} ({R.libelle(dr)}) ne peut pas être DR : {R.statut_dp(dr)}. "
                "Choisis un code plus précis (feuille), un autre code, ou null."), dr
    return None, dr


def run(llm, texts: list[str], R: Ressources, s: Settings = Settings(), log=print) -> list[dict]:
    """Code every text; returns one dict per text: dp, dr, justification, n_tools, forced, conv (the full
    conversation, JSON)."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=s.temperature, top_p=s.top_p, top_k=20, presence_penalty=s.presence_penalty,
                        max_tokens=s.max_tokens)
    tools = [TOOLS[x] for x in TOOL_NAMES]
    tok = llm.get_tokenizer()
    budget = s.max_model_len - s.max_tokens
    sysmsg = system_prompt(s.max_turns, s.procedure)
    n = len(texts)
    convs = [[{"role": "system", "content": sysmsg}, {"role": "user", "content": user_message(t)}] for t in texts]
    out = [dict(dp=None, dr=None, justification=None, n_tools=0, forced=False) for _ in texts]
    calls = [[] for _ in texts]  # (tool, args) already called
    finished, forced = [False] * n, [False] * n
    n_retry, n_force = [0] * n, [0] * n

    def length(conv: list[dict], with_tools: bool) -> int:
        text = tok.apply_chat_template(copy.deepcopy(conv), tools=tools if with_tools else None,
                                       add_generation_prompt=True, tokenize=False, enable_thinking=False)
        return len(tok.encode(text, add_special_tokens=False))

    def step(i: int, gen, last: bool) -> int:
        """Handle one generated message of conversation i; returns the number of tools called."""
        text = gen.text
        visible = THINK.sub("", text).strip()
        if gen.finish_reason == "length" and not last:  # cut by max_tokens: retry the turn, nothing kept
            n_retry[i] += 1
            finished[i] = n_retry[i] > 2
            return 0
        found = TOOL_CALL.findall(visible) if not last else []
        if found:
            parsed = [parse_call(c) for c in found[:3]]
            convs[i].append({"role": "assistant", "content": TOOL_CALL.sub("", visible).strip(),
                             "tool_calls": [{"type": "function", "function": {
                                 "name": name or "invalide", "arguments": json.dumps(a, ensure_ascii=False)}}
                                 for name, a in parsed]})
            for name, a in parsed:
                res = R.call(name, a) if name else "Appel d'outil illisible (JSON invalide)."
                convs[i].append({"role": "tool", "content": res})
                calls[i].append((name, a))
            out[i]["n_tools"] += len(parsed)
            return len(parsed)
        dp, dr, why_text = parse_answer(text)
        dp, dr = (R.canon(dp) if dp else dp), (R.canon(dr) if dr else dr)
        why, dr = check(R, dp, dr)
        missing = [t for t in REQUIRED if t not in {c for c, _ in calls[i]}]
        if why is None and missing and not last and n_force[i] < 4:
            n_force[i] += 1
            convs[i].append({"role": "assistant", "content": visible})
            convs[i].append({"role": "user", "content": "Procédure obligatoire non respectée : avant de répondre, "
                             "appelle " + ", ".join(missing) + " pour ce dossier (étapes b à f), puis donne ta "
                             "réponse JSON."})
            return 0
        final = dr or dp
        checked = any(c == "fiche_code" and R.canon(a.get("code", "")) == final for c, a in calls[i]) if final else False
        if why is None and not checked and not last and n_retry[i] < 2:
            why = (f"Avant de conclure, vérifie {final} avec fiche_code (notes d'exclusion, code plus précis) "
                   "et compare-le aux autres candidats plausibles.")
        convs[i].append({"role": "assistant", "content": visible})
        if why is None or last or n_retry[i] >= 2:
            ok = why is None or (dp and CODE_OK.match(dp) and dp in R.cim)
            out[i].update(dp=dp if ok else None, dr=dr if ok else None, justification=why_text if ok else None)
            finished[i] = True
        else:
            n_retry[i] += 1
            convs[i].append({"role": "user", "content": why})
        return 0

    for turn in range(s.max_turns + 2):
        active = [i for i in range(n) if not finished[i]]
        if not active:
            break
        for i in active:  # max turns reached, or the next turn would not fit: ask for the answer, no tools
            if not forced[i] and (turn >= s.max_turns or length(convs[i], True) > budget):
                forced[i] = out[i]["forced"] = True
                convs[i].append({"role": "user", "content": FORCE})
            if forced[i] and length(convs[i], False) > budget:  # not even room for the answer
                finished[i] = True
        n_calls = 0
        for with_tools in (True, False):
            group = [i for i in active if not finished[i] and forced[i] != with_tools]
            if not group:
                continue
            # copies: vLLM rewrites tool_call arguments (str -> dict) IN PLACE in the messages it gets
            gens = llm.chat([copy.deepcopy(convs[i]) for i in group], sp, tools=tools if with_tools else None,
                            chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
            for i, g in zip(group, gens):
                n_calls += step(i, g.outputs[0], last=not with_tools)
        log(f"turn {turn}: {len(active)} active, {n_calls} tool calls, {sum(finished)}/{n} done")

    for o, c in zip(out, convs):
        o["conv"] = json.dumps(c, ensure_ascii=False)
    return out
