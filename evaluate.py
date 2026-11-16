#!/usr/bin/env python3
"""Step 3: run classifiers over the held-out set and compare them.

    # Laya (run inside the Laya container on the Spark: see spark/run.sh predict)
    python evaluate.py predict --backend laya --checkpoint /out/hb-review --out /out/pred.laya.jsonl

    # Jev (hosted API; sends the test diffs to TypeSafe)
    export TYPESAFE_API_KEY=...
    python evaluate.py predict --backend jev --out out/pred.jev.jsonl

    # Side-by-side report
    python evaluate.py report laya=out/pred.laya.jsonl jev=out/pred.jev.jsonl --out out/report

Every backend answers the same unit inputs with the same questions. A change-level answer is
the highest answer among the change's units, so one risky unit escalates the whole change.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import append_jsonl, level, options, read_jsonl

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
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("Set TYPESAFE_API_KEY.")

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


def score(preds: dict, units: list[dict], changes: list[dict], questions: dict) -> dict:
    answered = [u for u in units if u["id"] in preds]
    lat = sorted(preds[u["id"]]["ms"] for u in answered)
    out = {
        "units": len(units), "units_answered": len(answered),
        "units_truncated": sum(bool(preds[u["id"]].get("truncated")) for u in answered),
        "latency_ms_mean": round(sum(lat) / len(lat), 2) if lat else None,
        "latency_ms_p95": lat[int(0.95 * (len(lat) - 1))] if lat else None,
        "changes": len(changes), "questions": {},
    }
    scored = [c for c in changes if c["units"] and all(i in preds for i in c["units"])]
    out["changes_scored"] = len(scored)
    for qid, q in questions.items():
        hits = [preds[u["id"]]["answers"][qid]["label"] == u["expected"][qid] for u in answered]
        conf = [(preds[u["id"]]["answers"][qid]["p"], h) for u, h in zip(answered, hits)
                if preds[u["id"]]["answers"][qid]["p"] is not None]
        gold = [level(q, c["expected"][qid]) for c in scored]
        pred = [max(level(q, preds[i]["answers"][qid]["label"]) for i in c["units"]) for c in scored]
        need = [(g, p) for g, p in zip(gold, pred) if g > 0]          # changes that need escalation
        top = len(options(q)) - 1
        need_top = [(g, p) for g, p in zip(gold, pred) if g == top]   # changes at the highest level
        out["questions"][qid] = {
            "unit_accuracy": sum(hits) / len(hits) if hits else None,
            "unit_ece": ece(conf),
            "change_accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else None,
            "change_under_called": sum(p < g for g, p in zip(gold, pred)),
            "change_over_called": sum(p > g for g, p in zip(gold, pred)),
            "escalation_needed": len(need),
            "escalation_missed": sum(p == 0 for _, p in need),
            "top_level_needed": len(need_top),
            "top_level_missed": sum(p < top for _, p in need_top),
        }
    return out


def fmt(x, pct=False) -> str:
    if x is None:
        return "n/a"
    return f"{x:.1%}" if pct else (f"{x:.3f}" if isinstance(x, float) else str(x))


def cmd_report(args) -> int:
    units = read_jsonl(args.data)
    changes = read_jsonl(args.changes)
    questions = units[0]["questions"]
    results = {}
    for spec in args.predictions:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"Give predictions as name=path, got {spec!r}")
        results[name] = score({r["id"]: r for r in read_jsonl(path)}, units, changes, questions)

    md = ["# Classifier comparison", "",
          f"Held-out set: {len(changes)} changes, {len(units)} units. "
          f"{sum(not c['units_complete'] for c in changes)} changes exceed the unit cap and are only partly classified.",
          "Labels are teacher-inferred, so every number is agreement with the teacher.", "",
          "| Backend | Units answered | Truncated inputs | Changes scored | Latency mean (ms/unit) | Latency p95 |",
          "|---|---|---|---|---|---|"]
    for name, r in results.items():
        md.append(f"| {name} | {r['units_answered']}/{r['units']} | {r['units_truncated']} | "
                  f"{r['changes_scored']}/{r['changes']} | {fmt(r['latency_ms_mean'])} | {fmt(r['latency_ms_p95'])} |")
    md += ["", "Escalation missed: the change needed more than the lowest option and the classifier chose the lowest.",
           "Top level missed: the change needed the highest option and the classifier chose lower.", "",
           "| Question | Backend | Unit accuracy | Unit ECE | Change accuracy | Under-called | Over-called | "
           "Escalation missed | Top level missed |",
           "|---|---|---|---|---|---|---|---|---|"]
    for qid in questions:
        for name, r in results.items():
            m = r["questions"][qid]
            md.append(
                f"| {qid} | {name} | {fmt(m['unit_accuracy'], True)} | {fmt(m['unit_ece'])} | "
                f"{fmt(m['change_accuracy'], True)} | {m['change_under_called']} | {m['change_over_called']} | "
                f"{m['escalation_missed']}/{m['escalation_needed']} | {m['top_level_missed']}/{m['top_level_needed']} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    out.with_suffix(".json").write_text(json.dumps(results, indent=1))
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
    p.set_defaults(fn=cmd_predict)
    r = sub.add_parser("report")
    r.add_argument("predictions", nargs="+", help="name=path/to/predictions.jsonl")
    r.add_argument("--data", default="data/test.jsonl")
    r.add_argument("--changes", default="data/test_changes.jsonl")
    r.add_argument("--out", default="out/report")
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
