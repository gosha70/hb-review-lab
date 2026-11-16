"""Runs repo_audit against a local fake GitHub API. No network access needed."""
import csv
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REVERTED = "a" * 40
RL = {"remaining": 5000, "resetAt": "2030-01-01T00:00:00Z"}


def pr(number, title, author="alice", reviews=(), paths=("src/app.py",), ci="SUCCESS", merge=None):
    return {
        "number": number, "title": title, "createdAt": f"2026-01-{number:02d}T00:00:00Z",
        "mergedAt": None, "state": "MERGED", "additions": 10, "deletions": 2, "changedFiles": len(paths),
        "author": {"login": author, "__typename": "Bot" if author.endswith("[bot]") else "User"},
        "baseRefOid": "b" * 40, "headRefOid": "c" * 40, "mergeCommit": {"oid": merge} if merge else None,
        "comments": {"totalCount": 0},
        "reviews": {"nodes": [
            {"state": s, "author": {"login": who, "__typename": t}, "comments": {"totalCount": n}}
            for who, t, s, n in reviews]},
        "files": {"nodes": [{"path": p} for p in paths]},
        "commits": {"nodes": [{"commit": {"statusCheckRollup": {"state": ci} if ci else None}}]},
    }


def commit(oid, headline, body="", pr_number=None, parents=1, ci=None):
    return {
        "oid": oid, "committedDate": "2026-02-01T00:00:00Z", "messageHeadline": headline,
        "messageBody": body, "additions": 3, "deletions": 1, "changedFilesIfAvailable": 1,
        "parents": {"totalCount": parents}, "author": {"user": {"login": "alice", "__typename": "User"}},
        "associatedPullRequests": {"nodes": [{"number": pr_number}] if pr_number else []},
        "statusCheckRollup": {"state": ci} if ci else None,
    }


PR_PAGES = [
    [pr(1, "Add login", reviews=[("bob", "User", "CHANGES_REQUESTED", 3)], paths=("src/auth/login.py",),
        merge=REVERTED),
     pr(2, "Tidy", reviews=[("alice", "User", "COMMENTED", 1), ("copilot", "Bot", "COMMENTED", 2)])],
    [pr(3, "Bump deps", author="dependabot[bot]", paths=("package.json",)),
     pr(4, "Fix crash", ci="FAILURE")],
]
COMMIT_PAGES = [
    [commit(REVERTED, "Add login (#1)", pr_number=1),
     commit("d" * 40, 'Revert "Add login"', body=f"This reverts commit {REVERTED}.")],
    [commit("e" * 40, "Merge branch x", parents=2),
     commit("f" * 40, "fix typo in migration")],
]


class Fake(BaseHTTPRequestHandler):
    calls = []
    fail_once = set()

    def log_message(self, *a):
        pass

    def _send(self, code, obj, headers=()):
        body = json.dumps(obj).encode()
        self.send_response(code)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        Fake.calls.append(("GET", self.path))
        sha = self.path.rsplit("/", 1)[-1]
        name = "db/migrations/001.sql" if sha == "f" * 40 else "README.md"
        self._send(200, {"files": [{"filename": name}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        q, v = req["query"], req["variables"]
        assert "mutation" not in q
        kind = "prs" if "pullRequests" in q else "commits"
        Fake.calls.append((kind, v.get("cursor")))
        if v.get("name") == "gone":
            return self._send(200, {"data": {"repository": None},
                                    "errors": [{"type": "NOT_FOUND", "message": "nope"}]})
        key = (kind, v.get("cursor"))
        if key in Fake.fail_once:
            Fake.fail_once.discard(key)
            return self._send(429, {}, [("Retry-After", "0")])
        if v.get("name") == "empty":
            repo = {"isPrivate": False, "isArchived": False, "primaryLanguage": None,
                    "pullRequests": {"totalCount": 0, "pageInfo": {"hasNextPage": False, "endCursor": None},
                                     "nodes": []},
                    "defaultBranchRef": None}
            return self._send(200, {"data": {"rateLimit": RL, "repository": repo}})
        pages = PR_PAGES if kind == "prs" else COMMIT_PAGES
        i = int(v["cursor"] or 0)
        page = {"totalCount": 4, "nodes": pages[i],
                "pageInfo": {"hasNextPage": i + 1 < len(pages), "endCursor": str(i + 1)}}
        if kind == "prs":
            repo = {"isPrivate": True, "isArchived": False, "primaryLanguage": {"name": "Python"},
                    "pullRequests": page}
        else:
            repo = {"defaultBranchRef": {"name": "main", "target": {"history": page}}}
        self._send(200, {"data": {"rateLimit": RL, "repository": repo}})


class AuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        os.environ["GITHUB_API_URL"] = f"http://127.0.0.1:{cls.srv.server_port}"
        os.environ["GITHUB_TOKEN"] = "test"
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(k, None)
        os.environ["NO_PROXY"] = "127.0.0.1"
        import repo_audit
        cls.mod = repo_audit
        repo_audit.API = os.environ["GITHUB_API_URL"]
        repo_audit.time.sleep = lambda s: None

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def run_audit(self, out, *repos):
        args = ["--out", out]
        for r in repos:
            args += ["--repo", r]
        self.assertEqual(self.mod.main(args), 0)

    def test_metrics_retry_and_skip(self):
        Fake.fail_once = {("prs", "1")}
        with tempfile.TemporaryDirectory() as out:
            self.run_audit(out, "me/app", "me/gone", "me/empty")
            rows = {r["repo"]: r for r in csv.DictReader(open(f"{out}/repos.csv"))}
            self.assertEqual(set(rows), {"me/app", "me/empty"})
            r = rows["me/app"]
            self.assertEqual(r["prs"], "4")
            self.assertEqual(r["prs_bot_authored"], "1")
            self.assertEqual(r["prs_human_review"], "1")       # self-review and bot don't count
            self.assertEqual(r["prs_changes_requested"], "1")
            self.assertEqual(r["prs_bot_review_only"], "1")
            self.assertEqual(r["changes_ci_failed"], "1")
            self.assertEqual(r["direct_commits"], "2")          # merge + PR commit excluded
            self.assertEqual(r["changes_reverted"], "1")
            self.assertEqual(r["reverts"], "1")
            self.assertEqual(r["stratum_security"], "1")
            self.assertEqual(r["stratum_data_migration"], "1")  # from REST file lookup
            self.assertEqual(r["stratum_dependencies"], "0")    # bot-authored excluded
            self.assertEqual(r["candidates"], "5")
            self.assertEqual(r["candidates_with_outcome_evidence"], "2")
            self.assertEqual(rows["me/empty"]["commits_default_branch"], "0")
            cands = [json.loads(l) for l in open(f"{out}/candidates.jsonl")]
            pr1 = next(c for c in cands if c["id"] == 1)
            self.assertTrue(pr1["observed"]["reverted"])
            self.assertIn("me/gone", open(f"{out}/summary.md").read())

    def test_resume_skips_finished_pages(self):
        with tempfile.TemporaryDirectory() as out:
            self.run_audit(out, "me/app")
            first = open(f"{out}/repos.csv").read()
            Fake.calls = []
            self.run_audit(out, "me/app")
            self.assertEqual(Fake.calls, [])
            self.assertEqual(open(f"{out}/repos.csv").read(), first)

    def test_resume_mid_repo(self):
        with tempfile.TemporaryDirectory() as out:
            self.run_audit(out, "me/app")
            full = open(f"{out}/repos.csv").read()
            state = json.load(open(f"{out}/state.json"))
            state["me/app"]["prs"] = {"cursor": "1", "done": False}  # as if stopped after page 1
            json.dump(state, open(f"{out}/state.json", "w"))
            Fake.calls = []
            self.run_audit(out, "me/app")
            self.assertEqual(Fake.calls, [("prs", "1")])
            self.assertEqual(open(f"{out}/repos.csv").read(), full)  # duplicate page deduplicated


if __name__ == "__main__":
    unittest.main()
