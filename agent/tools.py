"""Offline tools of the coding agent. Everything is read from local files (no network): the ICD-10
FR index and the passages of the ATIH MCO methodological guide (`index/`), plus the ICD-10 labels and
synonyms used by the search. They are shipped with the model, in its `resources/` folder.

Tools exposed to the model (OpenAI function schema in TOOLS):
  chercher_cim10(texte)   BM25 over labels, inclusion notes and synonyms -> best codes
  fiche_code(code)        label, PMSI status, inclusions / exclusions, parent, children
  guide_mco(question)     BM25 over passages of the ATIH MCO methodological guide 2026
  comparer_codes(code)    every code of the 3-character category side by side (DP status, inclusions,
                          exclusions): to choose the precise code
"""

import json
import re
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Okapi

WORD = re.compile(r"[a-z0-9]+")
STOP = set(
    "de la le les des du un une et ou a au aux en d l par pour sur sans avec dans est sont "
    "non autre autres precise precisee sai".split()
)


def norm_tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return [w for w in WORD.findall(t) if w not in STOP and len(w) > 1]


def norm_code(code: str) -> str:
    """Upper case, dots and spaces removed; the `+` of PMSI extensions (C189+8, B24+0) is KEPT:
    those codes are written with it in the index and in the gold."""
    return re.sub(r"[^A-Z0-9+]", "", str(code).upper())


class Ressources:
    def __init__(self, index_dir: Path, synonyms: Path | None = None, labels: Path | None = None):
        self.cim = {}
        with open(index_dir / "cim10.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self.cim[r["code"]] = r
        self.labels = set(json.loads(labels.read_text())) if labels else None
        # a model may drop the `+` of an extension (C1898 for C189+8): map it back when unambiguous
        self.alias = {c.replace("+", ""): c for c in self.cim if "+" in c and c.replace("+", "") not in self.cim}
        # one BM25 "document" per term, mapped to its code; a code is scored by its best term
        terms, codes, kinds = [], [], []
        for code, r in self.cim.items():
            if r.get("mco_had") == "3":  # not codable (non-leaf): never return it
                continue
            for t, kind in [(r.get("libelle") or "", "libellé")] + [(i, "inclusion") for i in r.get("inclusions", [])]:
                if t:
                    terms.append(t), codes.append(code), kinds.append(kind)
        if synonyms is not None:
            import pandas as pd

            import pyarrow.parquet as pq

            # one row = one term (`note_text`) and its code (`affection`, or `dp` in older files)
            col = "dp" if "dp" in pq.read_schema(synonyms).names else "affection"
            syn = pd.read_parquet(synonyms, columns=["note_text", col]).drop_duplicates().dropna()
            for t, c in zip(syn["note_text"], syn[col]):
                if c in self.cim:
                    terms.append(str(t)), codes.append(c), kinds.append("synonyme")
        self.terms, self.term_codes, self.term_kinds = terms, codes, kinds
        self.bm25 = BM25Okapi([norm_tokens(t) for t in terms])
        self.guide = [json.loads(line) for line in open(index_dir / "guide_mco.jsonl", encoding="utf-8")]
        self.guide_bm25 = BM25Okapi([norm_tokens(g["section"] + " " + g["texte"]) for g in self.guide])

    # ---------------------------------------------------------------- helpers
    def canon(self, code: str) -> str:
        c = norm_code(code)
        return c if c in self.cim else self.alias.get(c, c)

    def libelle(self, code: str) -> str:
        r = self.cim.get(self.canon(code))
        return (r or {}).get("libelle") or (r or {}).get("libelle_pmsi") or "?"

    def statut_dp(self, code: str) -> str:
        r = self.cim.get(self.canon(code))
        if r is None:
            return "code inconnu"
        if self.labels is not None and r["code"] not in self.labels:
            return "hors des codes prédictibles"
        return {"0": "autorisé en DP"}.get(r.get("mco_had"), r.get("mco_had_libelle") or "?")

    # ---------------------------------------------------------------- tools
    def chercher_cim10(self, texte: str, k: int = 10) -> str:
        q = norm_tokens(texte)
        if not q:
            return "Requête vide."
        scores = self.bm25.get_scores(q)
        best: dict[str, tuple[float, str, str]] = {}
        for i in scores.argsort()[::-1][: 40 * k]:
            if scores[i] <= 0:
                break
            c = self.term_codes[i]
            if c not in best or scores[i] > best[c][0]:
                best[c] = (float(scores[i]), self.terms[i], self.term_kinds[i])
            if len(best) >= k:
                break
        if not best:
            return f"Aucun code trouvé pour « {texte} »."
        lines = [f"Résultats pour « {texte} » :"]
        for c, (s, t, kind) in sorted(best.items(), key=lambda x: -x[1][0]):
            lines.append(f"- {c} : {self.libelle(c)} [{self.statut_dp(c)}] (trouvé par {kind} : « {t[:80]} »)")
        return "\n".join(lines)

    def fiche_code(self, code: str) -> str:
        c = self.canon(code)
        r = self.cim.get(c)
        if r is None:
            # maybe a prefix: list the codes below it
            sub = sorted(x for x in self.cim if x.startswith(c))[:20]
            if sub:
                return f"{code} n'est pas un code ; codes commençant par {c} :\n" + "\n".join(
                    f"- {x} : {self.libelle(x)} [{self.statut_dp(x)}]" for x in sub
                )
            return f"Code {code} inconnu de la CIM-10 FR."
        out = [f"{c} : {r.get('libelle')}", f"Statut : {self.statut_dp(c)}"]
        for key, title in [("inclusions", "Inclusions"), ("exclusions", "Exclusions"),
                           ("aide_codage", "Aide au codage"), ("definition", "Définition"), ("notes", "Notes")]:
            if r.get(key):
                out.append(f"{title} (notes de {r.get('notes_de')}) :")
                out += [f"  • {x[:300]}" for x in r[key][:12]]
        if r.get("parent"):
            p = self.canon(r["parent"])
            out.append(f"Parent : {r['parent']} {self.libelle(p) if p in self.cim else ''}".rstrip())
        if r.get("enfants"):
            out.append("Codes plus précis :")
            out += [f"  - {e} : {self.libelle(e)} [{self.statut_dp(e)}]" for e in r["enfants"][:20]]
        return "\n".join(out)

    def guide_mco(self, question: str, k: int = 3) -> str:
        q = norm_tokens(question)
        if not q:
            return "Question vide."
        scores = self.guide_bm25.get_scores(q)
        out = [f"Extraits du Guide méthodologique MCO 2026 pour « {question} » :"]
        for i in scores.argsort()[::-1][:k]:
            g = self.guide[i]
            out.append(f"--- p. {g['page']} — {g['section'][:90]}\n{g['texte'][:1100]}")
        return "\n".join(out)

    def comparer_codes(self, code: str) -> str:
        c = self.canon(code)
        cat = c[:3]
        codes = sorted(x for x in self.cim if x[:3] == cat and x != cat and not self.cim[x].get("enfants"))
        if not codes:
            return f"Aucune catégorie CIM-10 « {cat} »."
        out = [f"Catégorie {cat} : {self.libelle(cat)} — {len(codes)} codes"]
        for x in codes[:30]:
            r = self.cim[x]
            out.append(f"- {x} : {self.libelle(x)} [{self.statut_dp(x)}]")
            for key, title in [("inclusions", "inclut"), ("exclusions", "exclut")]:
                if r.get(key):
                    out.append(f"    {title} : " + " ; ".join(t[:80] for t in r[key][:3]))
        if len(codes) > 30:
            out.append(f"... {len(codes) - 30} autres codes : préciser avec fiche_code.")
        return "\n".join(out)

    def call(self, name: str, args: dict) -> str:
        try:
            if name == "chercher_cim10":
                return self.chercher_cim10(str(args.get("texte", "")))
            if name == "fiche_code":
                return self.fiche_code(str(args.get("code", "")))
            if name == "guide_mco":
                return self.guide_mco(str(args.get("question", "")))
            if name == "comparer_codes":
                return self.comparer_codes(str(args.get("code", "")))
        except Exception as e:  # a tool error is returned to the model, never raised
            return f"Erreur de l'outil {name} : {type(e).__name__}: {e}"
        return f"Outil inconnu : {name}"


def _fn(name, desc, param, pdesc):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {param: {"type": "string", "description": pdesc}}, "required": [param]}}}


TOOLS = {
    "chercher_cim10": _fn(
        "chercher_cim10",
        "Cherche dans la CIM-10 FR (libellés, notes d'inclusion, synonymes) les codes correspondant à un "
        "terme médical. Renvoie au plus 10 codes avec leur libellé et s'ils sont autorisés en DP.",
        "texte", "Terme ou expression médicale décrivant la pathologie."),
    "fiche_code": _fn(
        "fiche_code",
        "Fiche d'un code CIM-10 : libellé, autorisation en DP, notes d'inclusion et d'EXCLUSION, code "
        "parent et codes plus précis. À utiliser pour vérifier un code avant de le choisir.",
        "code", "Code CIM-10, avec ou sans point."),
    "guide_mco": _fn(
        "guide_mco",
        "Cherche dans le Guide méthodologique MCO 2026 de l'ATIH les règles de codage (choix du DP, "
        "situations cliniques, DR, séances, symptômes vs maladie, codes Z...).",
        "question", "Question ou mots-clés sur la règle de codage."),
}
TOOLS["comparer_codes"] = _fn(
    "comparer_codes",
    "Liste côte à côte tous les codes d'une catégorie CIM-10 à 3 caractères (statut DP, inclusions, "
    "exclusions), pour choisir le code précis entre codes voisins.",
    "code", "Un code ou une catégorie à 3 caractères.")
