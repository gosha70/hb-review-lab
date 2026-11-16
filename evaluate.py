#!/usr/bin/env python3
"""Step 3: run classifiers over the held-out set and compare them.

    # Laya (run inside the Laya container on the Spark: see spark/run.sh predict)
    python evaluate.py predict --backend laya --checkpoint /out/hb-review --out /out/pred.laya.jsonl

    # Jev (hosted API; sends the test diffs to TypeSafe)
    export TYPESAFE_API_KEY=...
    python evaluate.py predict --backend jev --allow-external typesafe --out out/pred.jev.jsonl

    # Side-by-side report
    python evaluate.py report laya=out/pred.laya.jsonl jev=out/pred.jev.jsonl --out out/report

    # Teacher-disagreement screen for potential bias (see label_second_teacher.py)
    python evaluate.py teachers claude=data/labels.jsonl openai=data/labels.openai.jsonl

Every backend answers the same unit inputs with the same questions. A change-level answer is
the highest answer among the change's units, so one risky unit escalates the whole change.

Backends are compared only on changes that are complete, that every backend answered in full,
and whose inputs no backend truncated. A change with an incomplete diff, or with a unit the
model could not read whole, is never scored as classified: it is reported as needing escalation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import allowed_repos, append_jsonl, level, load_questions, options, read_jsonl

JEV_URL = os.environ.get("JEV_URL", "https://api.typesafe.ai/v1/systemone")


def normalise(answer: dict, q: dict) -> dict:
    """{label, p} from a Laya/Jev answer; p is the probability of the reported label."""
    probs = answer.get("probabilities") or {}
    if q["type"] == "score":
        if probs:
            label = int(max(probs, key=probs.get))
        else:
            label = level(q, answer["score"])
        p = probs.get(str(label))
    else:
        label = answer["choice"]
        p = probs.get(label)
    return {"label": label, "p": p}


def predict_laya(rows, questions, args):
    import laya
    agent = laya.load(args.checkpoint, device=args.device)
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        t0 = time.perf_counter()
        results = agent.predict_batch([r["state"] for r in chunk], questions,
                                      batch_size=args.batch_size, max_len=args.max_len)
        ms = (time.perf_counter() - t0) * 1000 / len(chunk)
        for r, res in zip(chunk, results):
            yield r["id"], res["answers"], ms, bool(res.get("usage", {}).get("truncated"))


def predict_jev(rows, questions, args):
    if args.allow_external != "typesafe":
        sys.exit("This step sends the test diffs to TypeSafe's Jev API.\n"
                 "Re-run with --allow-external typesafe to confirm. Only repositories listed under "
                 "'typesafe' in the allowlist file are sent.")
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("Set TYPESAFE_API_KEY.")
    allowed = allowed_repos("typesafe", args.allowlist)
    blocked = sorted({r["tags"][0] for r in rows} - allowed)
    if blocked:
        print(f"Not on the typesafe allowlist, will not be sent: {', '.join(blocked)}", file=sys.stderr)
    rows = [r for r in rows if r["tags"][0] in allowed]
    sent = Path(args.out).with_suffix(".sent.jsonl")
    for r in rows:   # audit trail of what leaves this machine
        append_jsonl(sent, {"provider": "typesafe", "id": r["id"], "repo": r["tags"][0],
                            "sent_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def call(r):
        body = json.dumps({"state": r["state"], "questions": questions, "model": args.jev_model}).encode()
        for attempt in range(6):
            req = urllib.request.Request(JEV_URL, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    out = json.load(resp)
                return r["id"], out["answers"], (time.perf_counter() - t0) * 1000, False
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    sys.exit(f"Jev rejected the key (HTTP {e.code}).")
                if e.code not in (408, 429) and e.code < 500:
                    return r["id"], None, 0.0, False
                time.sleep(float(e.headers.get("Retry-After") or 2 ** attempt))
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2 ** attempt)
        return r["id"], None, 0.0, False

    with ThreadPoolExecutor(args.concurrency) as pool:
        yield from pool.map(call, rows)


def cmd_predict(args) -> int:
    rows = read_jsonl(args.data)
    if not rows:
        sys.exit(f"No rows in {args.data}")
    questions = rows[0]["questions"]
    done = {r["id"] for r in read_jsonl(args.out)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(done)} already answered, {len(todo)} to run on {args.backend}", file=sys.stderr)
    run = predict_laya if args.backend == "laya" else predict_jev
    failed = 0
    for rid, answers, ms, truncated in run(todo, questions, args):
        if answers is None:
            failed += 1
            continue
        try:
            norm = {qid: normalise(answers[qid], q) for qid, q in questions.items()}
        except (KeyError, ValueError, TypeError):
            failed += 1
            continue
        append_jsonl(args.out, {"id": rid, "answers": norm, "ms": round(ms, 2), "truncated": truncated})
    print(f"Wrote {args.out}; {failed} units failed", file=sys.stderr)
    return 0


# ---------- report ----------

def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float | None:
    """Expected calibration error over (confidence, correct) pairs."""
    if not pairs:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inb = [(p, c) for p, c in pairs if lo < p <= hi or (b == 0 and p == 0)]
        if inb:
            total += abs(sum(p for p, _ in inb) / len(inb) - sum(c for _, c in inb) / len(inb)) * len(inb)
    return total / len(pairs)


def score(preds: dict, units: list[dict], changes: list[dict], scored: list[dict], questions: dict) -> dict:
    answered = [u for u in units if u["id"] in preds]
    lat = sorted(preds[u["id"]]["ms"] for u in answered)
    n_truncated = sum(bool(preds[u["id"]].get("truncated")) for u in answered)
    # Accuracy is measured only where the model read the whole input.
    answered = [u for u in answered if not preds[u["id"]].get("truncated")]
    out = {
        "units": len(units), "units_answered": len(answered) + n_truncated,
        "units_truncated": n_truncated,
        "changes_truncated": sum(any(preds.get(i, {}).get("truncated") for i in c["units"]) for c in changes),
        "latency_ms_mean": round(sum(lat) / len(lat), 2) if lat else None,
        "latency_ms_p95": lat[int(0.95 * (len(lat) - 1))] if lat else None,
        "changes": len(changes), "questions": {},
    }
    out["changes_answered"] = sum(bool(c["units"]) and all(i in preds for i in c["units"]) for c in changes)
    out["changes_scored"] = len(scored)
    authors = sorted({c.get("written_by", "unmarked") for c in scored})
    out["by_author"] = {}
    for qid, q in questions.items():
        hits = [preds[u["id"]]["answers"][qid]["label"] == u["expected"][qid] for u in answered]
        conf = [(preds[u["id"]]["answers"][qid]["p"], h) for u, h in zip(answered, hits)
                if preds[u["id"]]["answers"][qid]["p"] is not None]
        out["questions"][qid] = {
            "unit_accuracy": sum(hits) / len(hits) if hits else None,
            "unit_ece": ece(conf),
            **change_metrics(scored, qid, q, unit_max(preds, qid, q)),
        }
        for a in authors:
            group = [c for c in scored if c.get("written_by", "unmarked") == a]
            out["by_author"].setdefault(a, {})[qid] = {
                "changes": len(group), **change_metrics(group, qid, q, unit_max(preds, qid, q))}
    return out


def unit_max(preds: dict, qid: str, q: dict):
    """A backend's change-level answer: the highest answer among the change's units."""
    return lambda c: max(level(q, preds[i]["answers"][qid]["label"]) for i in c["units"])


def majority_baseline(maj: dict, units: list[dict], scored: list[dict], questions: dict) -> dict:
    """A classifier that always gives the train split's most common answer.

    Unit accuracy uses the unit majority. Change-level numbers use the change majority, which
    can differ: a change takes the highest of its units, so most units can be the lowest option
    while most changes are not.
    """
    out = {"questions": {}, "by_author": {}}
    authors = sorted({c.get("written_by", "unmarked") for c in scored})
    for qid, q in questions.items():
        hits = [u["expected"][qid] == maj["unit"][qid] for u in units]
        const = (lambda lv: lambda c: lv)(level(q, maj["change"][qid]))
        out["questions"][qid] = {
            "unit_accuracy": sum(hits) / len(hits) if hits else None, "unit_ece": None,
            "unit_answer": maj["unit"][qid], "change_answer": maj["change"][qid],
            **change_metrics(scored, qid, q, const),
        }
        for a in authors:
            group = [c for c in scored if c.get("written_by", "unmarked") == a]
            out["by_author"].setdefault(a, {})[qid] = {"changes": len(group), **change_metrics(group, qid, q, const)}
    return out


def change_metrics(group: list[dict], qid: str, q: dict, pred_level) -> dict:
    gold = [level(q, c["expected"][qid]) for c in group]
    pred = [pred_level(c) for c in group]
    need = [(g, p) for g, p in zip(gold, pred) if g > 0]          # changes that need escalation
    top = len(options(q)) - 1
    need_top = [(g, p) for g, p in zip(gold, pred) if g == top]   # changes at the highest level
    return {
        "change_accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else None,
        "change_under_called": sum(p < g for g, p in zip(gold, pred)),
        "change_over_called": sum(p > g for g, p in zip(gold, pred)),
        "escalation_needed": len(need),
        "escalation_missed": sum(p == 0 for _, p in need),
        "top_level_needed": len(need_top),
        "top_level_missed": sum(p < top for _, p in need_top),
    }


def cmd_teachers(args) -> int:
    """Where two teachers disagree, and in which direction, split by who wrote the change.

    A screen for potential bias, not proof of it: the author groups are not matched.
    """
    questions = load_questions(args.questions)
    (na, pa), (nb, pb) = (s.partition("=")[::2] for s in (args.a, args.b))
    la = {r["case_id"]: r["change"] for r in read_jsonl(pa)}
    lb = {r["case_id"]: r["change"] for r in read_jsonl(pb)}
    author = {c["case_id"]: c.get("written_by", "unmarked") for c in read_jsonl(args.cases)}
    shas = {r.get("questions_sha") for p in (pa, pb) for r in read_jsonl(p)}
    if len(shas) > 1:
        sys.exit("The two label files were made with different questions.json versions; they cannot be compared.")
    both = sorted(set(la) & set(lb))
    if not both:
        sys.exit("The two label files share no cases.")
    md = ["# Teacher disagreement (potential-bias screen)", "",
          "This shows where two teachers disagree and in which direction. It is a screen, not a test "
          "of self-preference: a gap can also come from ordinary differences between the models or "
          "between the groups of changes, which are not matched. Confirming bias needs matched cases "
          "and independently reviewed labels.", "",
          f"{len(both)} changes labelled by both {na} and {nb}.",
          f"'{na} lower' means {na} chose a less severe answer than {nb} for the same change.", "",
          f"| Question | Written by | Changes | Agree | {na} lower | {na} higher |", "|---|---|---|---|---|---|"]
    result = {}
    for qid, q in questions.items():
        for a in sorted({author.get(k, "unmarked") for k in both}):
            ids = [k for k in both if author.get(k, "unmarked") == a]
            d = [level(q, la[k][qid]) - level(q, lb[k][qid]) for k in ids]
            row = {"changes": len(ids), "agree": sum(x == 0 for x in d) / len(d),
                   "a_lower": sum(x < 0 for x in d) / len(d), "a_higher": sum(x > 0 for x in d) / len(d)}
            result.setdefault(qid, {})[a] = row
            md.append(f"| {qid} | {a} | {len(ids)} | {row['agree']:.1%} | {row['a_lower']:.1%} | {row['a_higher']:.1%} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    print("\n".join(md))
    return 0


def fmt(x, pct=False) -> str:
    if x is None:
        return "n/a"
    return f"{x:.1%}" if pct else (f"{x:.3f}" if isinstance(x, float) else str(x))


def cmd_report(args) -> int:
    units = read_jsonl(args.data)
    changes = read_jsonl(args.changes)
    # With no units at all (a test set of only unreadable changes) there is nothing to score,
    # but the coverage section still shows that every change requires escalation.
    questions = units[0]["questions"] if units else load_questions()
    preds = {}
    for spec in args.predictions:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"Give predictions as name=path, got {spec!r}")
        preds[name] = {r["id"]: r for r in read_jsonl(path)}
    incomplete = [c for c in changes if not c["units_complete"]]
    complete = [c for c in changes if c["units_complete"] and c["units"]]
    truncated = [c for c in complete
                 if any(p.get(i, {}).get("truncated") for p in preds.values() for i in c["units"])]
    scored = [c for c in complete if c not in truncated
              and all(i in p for p in preds.values() for i in c["units"])]
    reasons = Counter(r for c in incomplete for r in c.get("incomplete_reasons") or ["unspecified"])
    results = {name: score(p, units, changes, scored, questions) for name, p in preds.items()}
    # Constant baseline from the train split, shown in the per-question tables only.
    rows = dict(results)
    maj_path = Path(args.majority) if args.majority else Path(args.data).parent / "majority.json"
    maj = json.loads(maj_path.read_text()) if maj_path.exists() else None
    if maj and all(maj["unit"].get(q) is not None and maj["change"].get(q) is not None for q in questions):
        rows["majority"] = majority_baseline(maj, units, scored, questions)

    md = ["# Classifier comparison", "",
          f"Held-out set: {len(changes)} changes, {len(units)} units.",
          f"- {len(scored)} changes are scored: complete diff, answered in full by every backend, no truncated input.",
          f"- {len(incomplete)} changes have an incomplete diff"
          + (f" ({', '.join(f'{k}: {v}' for k, v in sorted(reasons.items()))})" if reasons else "")
          + ". Not scored; they require escalation.",
          f"- {len(truncated)} complete changes had a unit that a backend truncated. Not scored; they require escalation.",
          f"- {len(complete) - len(scored) - len(truncated)} complete changes were not answered by every backend.",
          "", "Labels are teacher-inferred, so every number is agreement with the teacher.", "",
          "Unit accuracy and ECE exclude units whose input the backend truncated.", "",
          "| Backend | Units answered | Truncated units | Changes with a truncated unit | Changes answered | "
          "Latency mean (ms/unit) | Latency p95 |",
          "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        md.append(f"| {name} | {r['units_answered']}/{r['units']} | {r['units_truncated']} | {r['changes_truncated']} | "
                  f"{r['changes_answered']}/{r['changes']} | {fmt(r['latency_ms_mean'])} | {fmt(r['latency_ms_p95'])} |")
    md += ["", "Escalation missed: the change needed more than the lowest option and the classifier chose the lowest.",
           "Top level missed: the change needed the highest option and the classifier chose lower.", "",
           "| Question | Backend | Unit accuracy | Unit ECE | Change accuracy | Under-called | Over-called | "
           "Escalation missed | Top level missed |",
           "|---|---|---|---|---|---|---|---|---|"]
    for qid in questions:
        for name, r in rows.items():
            m = r["questions"][qid]
            md.append(
                f"| {qid} | {name} | {fmt(m['unit_accuracy'], True)} | {fmt(m['unit_ece'])} | "
                f"{fmt(m['change_accuracy'], True)} | {m['change_under_called']} | {m['change_over_called']} | "
                f"{m['escalation_missed']}/{m['escalation_needed']} | {m['top_level_missed']}/{m['top_level_needed']} |")
    if "majority" in rows:
        md += ["", "`majority` always gives the train split's most common answer (unit majority for unit "
               "accuracy, change majority for the change columns). A backend that does not beat it on a "
               "question has learned nothing useful there."]
    md += ["", "## Teacher labels on the scored changes", "", "| Question | Changes per answer |", "|---|---|"]
    for qid, q in questions.items():
        dist = Counter(c["expected"][qid] for c in scored)
        md.append(f"| {qid} | " + ", ".join(f"{o}: {dist[o]}" for o in options(q)) + " |")
    groups = Counter(c.get("written_by", "unmarked") for c in scored)
    if len(groups) > 1:
        md += ["", "## By author", "",
               "'unmarked' means no AI-agent marker was found, not that a human wrote it. "
               "'Teacher escalates' is the share the teacher labelled above the lowest option.", "",
               "| Question | Written by | Changes | Teacher escalates | Backend | Change accuracy | Under-called | "
               "Escalation missed |", "|---|---|---|---|---|---|---|---|"]
        for qid in questions:
            for a in sorted(groups):
                for name, r in rows.items():
                    m = r["by_author"][a][qid]
                    md.append(f"| {qid} | {a} | {m['changes']} | {fmt(m['escalation_needed'] / m['changes'], True)} | "
                              f"{name} | {fmt(m['change_accuracy'], True)} | {m['change_under_called']} | "
                              f"{m['escalation_missed']}/{m['escalation_needed']} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    print("\n".join(md))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--backend", choices=["laya", "jev"], required=True)
    p.add_argument("--data", default="data/test.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default="english", help="laya: checkpoint directory or name")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--jev-model", default="jev-1.13.0")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--allow-external", choices=["typesafe"],
                   help="jev: required; confirms the test diffs may be sent to TypeSafe")
    p.add_argument("--allowlist", default="external_allowlist.json",
                   help="jev: JSON file naming the repositories whose code may be sent to each provider")
    p.set_defaults(fn=cmd_predict)
    r = sub.add_parser("report")
    r.add_argument("predictions", nargs="+", help="name=path/to/predictions.jsonl")
    r.add_argument("--data", default="data/test.jsonl")
    r.add_argument("--changes", default="data/test_changes.jsonl")
    r.add_argument("--majority", default=None,
                   help="baseline file from build_dataset.py (default: majority.json next to --data)")
    r.add_argument("--out", default="out/report")
    r.set_defaults(fn=cmd_report)
    t = sub.add_parser("teachers", help="compare two teachers' change-level labels")
    t.add_argument("a", help="name=path/to/labels.jsonl")
    t.add_argument("b", help="name=path/to/labels.jsonl")
    t.add_argument("--cases", default="data/cases.jsonl")
    t.add_argument("--questions", default=None)
    t.add_argument("--out", default="out/teachers")
    t.set_defaults(fn=cmd_teachers)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
