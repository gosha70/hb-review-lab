"""Offline tests for scrape -> label parsing -> dataset build -> report. No network, no models."""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import build_dataset
import common
import evaluate
import label_changes

PATCH = "@@ -1,2 +1,3 @@\n context\n+token = 'abcdefgh12345678'\n" + "@@ -9,1 +10,2 @@\n+more\n" * 3
OBS = {"human_reviews": 1, "human_inline_comments": 1, "changes_requested": True, "bot_reviews": 0,
       "discussion_comments": 0, "ci_failed": False, "reverted": False, "is_revert": False, "is_fix": False}


def cand(kind, id_, date, **kw):
    return {"repo": "me/app", "kind": kind, "id": id_, "title": f"Change {id_}", "date": date,
            "state": "MERGED" if kind == "pr" else "COMMITTED", "base_sha": "b" * 40, "head_sha": "h" * 40,
            "author": "alice", "author_is_bot": False, "strata": [], "observed": dict(OBS), **kw}


class FakeGitHub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.path.split("?")[0]
        f = [{"filename": "src/auth.py", "status": "modified", "additions": 4, "deletions": 0, "patch": PATCH},
             {"filename": "logo.png", "status": "added", "additions": 0, "deletions": 0}]
        if p.endswith("/pulls/1"):
            body = {"body": "Adds login"}
        elif p.endswith("/reviews"):
            body = [{"state": "CHANGES_REQUESTED", "user": {"login": "bob", "type": "User"},
                     "commit_id": "f" * 40, "submitted_at": "2026-01-02T00:00:00Z", "body": "unsafe"}]
        elif p.endswith("/comments"):
            body = [{"path": "src/auth.py", "line": 2, "user": {"login": "bob", "type": "User"},
                     "body": "hard-coded token"}]
        elif "/compare/" in p:
            body = {"files": f}
        elif p.endswith("/files"):
            body = f
        elif "/commits/" in p:
            body = {"commit": {"message": "Tweak\n\nbody text"}, "files": f[:1]}
        else:
            self.send_response(404)
            self.end_headers()
            return
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def teacher_answer(units, change_sec="specialist", unit_sec="specialist"):
    base = {"risk_score": "3", "criticality": "high", "human_review": "required", "ux_review": "none",
            "performance_review": "none"}
    return json.dumps({
        "rationale": "auth change",
        "change": {**base, "security_review": change_sec},
        "units": [{"unit": u["unit"], **base, "security_review": unit_sec} for u in units],
    })


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), FakeGitHub)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(k, None)
        os.environ.update(GITHUB_TOKEN="t", NO_PROXY="127.0.0.1",
                          GITHUB_API_URL=f"http://127.0.0.1:{cls.srv.server_port}")
        import repo_audit
        repo_audit.API = os.environ["GITHUB_API_URL"]
        import scrape_changes
        cls.scrape = scrape_changes

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_units_cover_patch_and_flag_overflow(self):
        case = {"repo": "r", "title": "t", "body": "", "files": [
            {"path": "a.py", "status": "modified", "additions": 1, "deletions": 0, "patch": PATCH}]}
        units, reasons = common.build_units(case, unit_chars=60, max_units=40)
        self.assertEqual(reasons, [])
        self.assertGreater(len(units), 1)
        joined = "".join(u["state"].split("Diff:\n", 1)[1] for u in units)
        self.assertEqual(joined, PATCH)                    # nothing dropped
        _, reasons = common.build_units(case, unit_chars=60, max_units=1)
        self.assertEqual(reasons, ["unit_cap"])

    def test_long_lines_are_kept_whole_across_units(self):
        patch = "@@ -1 +1 @@\n+" + "x" * 500 + "\n+tail\n"
        case = {"repo": "r", "title": "t", "body": "", "files": [
            {"path": "a.min.js", "status": "modified", "additions": 2, "deletions": 0, "patch": patch}]}
        units, reasons = common.build_units(case, unit_chars=100, max_units=40)
        self.assertEqual("".join(u["state"].split("Diff:\n", 1)[1] for u in units), patch)
        self.assertEqual(reasons, [])

    def test_missing_diffs_make_a_change_incomplete(self):
        ok = {"path": "a.py", "status": "modified", "additions": 1, "deletions": 0, "patch": "@@ -1 +1 @@\n+x\n"}
        big = {"path": "big.sql", "status": "modified", "additions": 9000, "deletions": 0, "patch": None}
        binary = {"path": "logo.png", "status": "added", "additions": 0, "deletions": 0, "patch": None}
        base = {"repo": "r", "title": "t", "body": ""}
        self.assertEqual(common.build_units({**base, "files": [ok, binary]})[1], [])
        self.assertEqual(common.build_units({**base, "files": [ok, big]})[1], ["file_without_diff"])
        self.assertEqual(common.build_units({**base, "files": [ok], "files_total": 400})[1], ["files_not_fetched"])

    def test_redact(self):
        out = common.redact("password = 'hunter2hunter2'\nkey AKIAABCDEFGHIJKLMNOP\nx = 1")
        self.assertNotIn("hunter2", out)
        self.assertNotIn("AKIAABCDEFGHIJKLMNOP", out)
        self.assertIn("x = 1", out)

    def test_end_to_end(self):
        questions = common.load_questions()
        with tempfile.TemporaryDirectory() as d:
            cands = [cand("pr", 1, "2026-01-01T00:00:00Z", body_edited_at="2026-01-03T00:00:00Z"),
                     cand("commit", "c" * 40, "2026-02-01T00:00:00Z"),
                     cand("commit", "d" * 40, "2026-03-01T00:00:00Z"),
                     cand("commit", "e" * 40, "2026-04-01T00:00:00Z", author_is_bot=True)]
            common.write_jsonl(f"{d}/candidates.jsonl", cands)
            argv = ["--candidates", f"{d}/candidates.jsonl", "--out", f"{d}/cases.jsonl"]
            self.assertEqual(self.scrape.main(argv), 0)
            self.assertEqual(self.scrape.main(argv), 0)    # resume adds nothing
            cases = common.read_jsonl(f"{d}/cases.jsonl")
            self.assertEqual(len(cases), 3)                # bot-authored change excluded
            pr = next(c for c in cases if c["kind"] == "pr")
            self.assertEqual(pr["diff_source"], "at_first_review")
            self.assertEqual((pr["body"], pr["body_source"]), ("", "omitted_edited_after_first_review"))
            self.assertNotIn("abcdefgh12345678", pr["files"][0]["patch"])
            self.assertEqual(pr["outcome"]["review_comments"][0]["body"], "hard-coded token")
            self.assertNotIn("outcome", json.dumps({k: v for k, v in pr.items() if k != "outcome"}))

            # teacher request and response handling
            p = label_changes.params(pr, questions, "m", 1800, 40)
            self.assertIn("hard-coded token", p["messages"][0]["content"])
            self.assertEqual(p["output_config"]["format"]["schema"]["properties"]["change"]["properties"]
                             ["security_review"]["enum"], ["none", "automated", "specialist"])
            for c in cases:
                units, _ = common.build_units(c)
                lab = label_changes.parse(teacher_answer(units), c, questions, 1800, 40)
                self.assertEqual(lab["change"]["risk_score"], 3)
                common.append_jsonl(f"{d}/labels.jsonl", {"case_id": c["case_id"], "unit_chars": 1800,
                                                          "max_units": 40, **lab})
            with self.assertRaises(ValueError):
                label_changes.parse(teacher_answer([]), pr, questions, 1800, 40)
            with self.assertRaises(SystemExit):      # nothing is sent without explicit confirmation
                label_changes.main(["--cases", f"{d}/cases.jsonl", "--out", f"{d}/x.jsonl"])

            # dataset: newest change of the repo is held out; rows are in Laya's format
            self.assertEqual(build_dataset.main(["--cases", f"{d}/cases.jsonl", "--labels", f"{d}/labels.jsonl",
                                                 "--out", d, "--test-frac", "0.34"]), 0)
            train, test = common.read_jsonl(f"{d}/train.jsonl"), common.read_jsonl(f"{d}/test.jsonl")
            val = common.read_jsonl(f"{d}/val.jsonl")
            self.assertEqual({r["case_id"] for r in val}, {"me/app@" + "c" * 12})
            with self.assertRaises(SystemExit):
                evaluate.main(["predict", "--backend", "jev", "--data", f"{d}/test.jsonl", "--out", f"{d}/j.jsonl"])
            self.assertFalse(os.path.exists(f"{d}/j.sent.jsonl"))
            self.assertEqual({r["case_id"] for r in test}, {"me/app@" + "d" * 12})
            self.assertFalse({r["case_id"] for r in train} & {r["case_id"] for r in test})
            self.assertEqual(set(train[0]) >= {"state", "questions", "expected"}, True)
            self.assertNotIn("hard-coded token", train[0]["state"])   # outcome never reaches the input
            self.assertEqual(train[0]["expected"]["security_review"], "specialist")

            # report: backend A agrees with the teacher, backend B misses the escalation
            def preds(path, sec):
                for r in test:
                    ans = {q: {"label": v, "p": 0.9} for q, v in r["expected"].items()}
                    ans["security_review"] = {"label": sec, "p": 0.9}
                    common.append_jsonl(path, {"id": r["id"], "answers": ans, "ms": 5.0})
            preds(f"{d}/a.jsonl", "specialist")
            preds(f"{d}/b.jsonl", "none")
            evaluate.main(["report", f"a={d}/a.jsonl", f"b={d}/b.jsonl", "--data", f"{d}/test.jsonl",
                           "--changes", f"{d}/test_changes.jsonl", "--out", f"{d}/report"])
            rep = json.load(open(f"{d}/report.json"))
            self.assertEqual(rep["a"]["questions"]["security_review"]["escalation_missed"], 0)
            self.assertEqual(rep["a"]["questions"]["security_review"]["change_accuracy"], 1.0)
            b = rep["b"]["questions"]["security_review"]
            self.assertEqual((b["escalation_missed"], b["escalation_needed"]), (1, 1))
            self.assertEqual((b["top_level_missed"], b["change_under_called"]), (1, 1))
            self.assertIn("| security_review | b |", open(f"{d}/report.md").read())

            # a backend that skipped the change, or an incomplete diff, leaves nothing to score
            evaluate.main(["report", f"a={d}/a.jsonl", f"none={d}/missing.jsonl", "--data", f"{d}/test.jsonl",
                           "--changes", f"{d}/test_changes.jsonl", "--out", f"{d}/report2"])
            self.assertEqual(json.load(open(f"{d}/report2.json"))["a"]["changes_scored"], 0)
            changes = common.read_jsonl(f"{d}/test_changes.jsonl")
            changes[0]["units_complete"] = False
            common.write_jsonl(f"{d}/test_changes.jsonl", changes)
            evaluate.main(["report", f"b={d}/b.jsonl", "--data", f"{d}/test.jsonl",
                           "--changes", f"{d}/test_changes.jsonl", "--out", f"{d}/report3"])
            b3 = json.load(open(f"{d}/report3.json"))["b"]
            self.assertEqual((b3["changes_scored"], b3["questions"]["security_review"]["escalation_needed"]), (0, 0))
            self.assertIn("1 changes have an incomplete diff", open(f"{d}/report3.md").read())

    def test_normalise_answers(self):
        q = common.load_questions()
        self.assertEqual(evaluate.normalise({"choice": "automated", "probabilities": {"automated": 0.7}},
                                            q["security_review"]), {"label": "automated", "p": 0.7})
        self.assertEqual(evaluate.normalise({"score": 2.6, "probabilities": {"2": 0.3, "3": 0.6}},
                                            q["risk_score"]), {"label": 3, "p": 0.6})
        self.assertEqual(evaluate.normalise({"score": 3.6}, q["risk_score"])["label"], 4)


if __name__ == "__main__":
    unittest.main()
