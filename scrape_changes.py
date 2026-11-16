#!/usr/bin/env python3
"""Step 1b: fetch diffs and review outcomes for the changes found by repo_audit.py.

Reads audit_out/candidates.jsonl and writes one case per change, with the two views kept apart:

  prediction view  title, body, files[] with patches   (what a classifier may see)
  outcome view     outcome{} with review comments, review states, CI, reverts
                   (hindsight: for the teacher and for analysis only)

For a reviewed pull request the prediction view is the diff as it stood at the first human
review, when GitHub still has that commit; otherwise the final diff is used and
diff_source says so. Secrets are redacted before anything is written.

Usage:
    export GITHUB_TOKEN=...
    python scrape_changes.py --candidates audit_out/candidates.jsonl --out data/cases.jsonl

Read-only (GET requests). Re-running resumes.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
from collections import defaultdict

import repo_audit as gh
from common import append_jsonl, read_jsonl, redact, written_by

MAX_FILE_PAGES = 3  # 300 files
SINGLE_CALL_FILE_CAP = 300  # compare and commit endpoints return at most this many files


def get(token: str, path: str):
    return gh._request(f"{gh.API}{path}", token)


def paged(token: str, path: str, max_pages: int = 10) -> list:
    out = []
    for page in range(1, max_pages + 1):
        sep = "&" if "?" in path else "?"
        rows = get(token, f"{path}{sep}per_page=100&page={page}")
        out += rows
        if len(rows) < 100:
            break
    return out


def files_of(rows: list) -> list[dict]:
    return [{
        "path": f["filename"], "status": f.get("status", "modified"),
        "additions": f.get("additions", 0), "deletions": f.get("deletions", 0),
        "patch": redact(f.get("patch")),   # None for binary or very large files
    } for f in rows]


def case_id(c: dict) -> str:
    return f"{c['repo']}#{c['id']}" if c["kind"] == "pr" else f"{c['repo']}@{c['id'][:12]}"


def scrape_pr(token: str, c: dict) -> dict:
    repo, n = c["repo"], c["id"]
    pr = get(token, f"/repos/{repo}/pulls/{n}")
    reviews = paged(token, f"/repos/{repo}/pulls/{n}/reviews")
    comments = paged(token, f"/repos/{repo}/pulls/{n}/comments")
    commits = paged(token, f"/repos/{repo}/pulls/{n}/commits", 3)
    author = written_by(c["author"], pr.get("body"), *[k["commit"]["message"] for k in commits])

    def human(u):
        return u and u.get("type") != "Bot" and u.get("login") != c["author"]

    # Three-dot compare diffs from the merge base of the two commits, i.e. the point where the PR
    # branched off, so the result is the PR's own changes as of the first review even if the base
    # branch has moved since. If that commit is gone (force-push), fall back to the final diff.
    files, source = None, "final"
    first = next((r for r in sorted(reviews, key=lambda r: r.get("submitted_at") or "")
                  if human(r.get("user")) and r.get("commit_id")), None)
    if first and first["commit_id"] != c["head_sha"]:
        try:
            cmp_ = get(token, f"/repos/{repo}/compare/{c['base_sha']}...{first['commit_id']}")
            files, source = files_of(cmp_.get("files", [])), "at_first_review"
            # Completeness is judged against this historical revision, not the final PR. The
            # endpoint gives no file total, so a full page means there may be more.
            files_complete = len(files) < SINGLE_CALL_FILE_CAP
        except (gh.RepoUnavailable, urllib.error.HTTPError):
            source = "final_first_review_commit_unavailable"
    if files is None:
        files = files_of(paged(token, f"/repos/{repo}/pulls/{n}/files", MAX_FILE_PAGES))
        files_complete = len(files) >= (pr.get("changed_files") or 0)   # same (final) revision

    # GitHub keeps only the current description. If it was edited after the first review it may
    # describe the fixes, so it is left out of the prediction view.
    body, body_source = pr.get("body"), "current"
    edited = c.get("body_edited_at")
    if first and edited and edited > (first.get("submitted_at") or ""):
        body, body_source = "", "omitted_edited_after_first_review"

    return {
        "body": redact(body), "body_source": body_source, "files": files, "diff_source": source,
        "files_complete": files_complete, "written_by": author,
        "outcome": {
            "observed": c["observed"],
            "reviews": [{"state": r["state"], "author": (r.get("user") or {}).get("login"),
                         "human": bool(human(r.get("user"))), "body": redact(r.get("body"))}
                        for r in reviews if r.get("body") or r["state"] != "COMMENTED"],
            "review_comments": [{"path": m.get("path"), "line": m.get("line") or m.get("original_line"),
                                 "author": (m.get("user") or {}).get("login"),
                                 "human": bool(human(m.get("user"))), "body": redact(m.get("body"))}
                                for m in comments],
        },
    }


def scrape_commit(token: str, c: dict) -> dict:
    d = get(token, f"/repos/{c['repo']}/commits/{c['id']}")
    message = d["commit"]["message"]
    body = message.split("\n", 1)[1].strip() if "\n" in message else ""
    return {
        "body": redact(body), "body_source": "current",
        "files": files_of(d.get("files", [])), "diff_source": "final",
        "files_complete": len(d.get("files", [])) < SINGLE_CALL_FILE_CAP,
        "written_by": written_by(c["author"], message),
        "outcome": {"observed": c["observed"], "reviews": [], "review_comments": []},
    }


def select(cands: list[dict], max_per_repo: int, include_unmerged: bool) -> list[dict]:
    by_repo = defaultdict(list)
    for c in cands:
        # Bots such as dependency updaters are dropped; AI coding agents are kept.
        if c["observed"]["is_revert"] or (c["author_is_bot"] and written_by(c["author"]) == "unmarked"):
            continue
        if c["kind"] == "pr" and c["state"] != "MERGED" and not include_unmerged:
            continue
        by_repo[c["repo"]].append(c)
    out = []
    for rows in by_repo.values():
        rows.sort(key=lambda c: c["date"], reverse=True)
        out += rows[:max_per_repo] if max_per_repo else rows
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidates", default="audit_out/candidates.jsonl")
    ap.add_argument("--out", default="data/cases.jsonl")
    ap.add_argument("--max-per-repo", type=int, default=0, help="newest N changes per repository (0 = all)")
    ap.add_argument("--include-unmerged", action="store_true", help="also take closed, unmerged PRs")
    args = ap.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN to a token with read access to the repositories.")

    done = {r["case_id"] for r in read_jsonl(args.out)}
    todo = [c for c in select(read_jsonl(args.candidates), args.max_per_repo, args.include_unmerged)
            if case_id(c) not in done]
    gh.log(f"{len(done)} cases already scraped, {len(todo)} to fetch")
    written = skipped = 0
    for i, c in enumerate(todo, 1):
        try:
            detail = scrape_pr(token, c) if c["kind"] == "pr" else scrape_commit(token, c)
        except (gh.RepoUnavailable, urllib.error.HTTPError) as e:
            gh.log(f"  skipped {case_id(c)}: {e}")
            skipped += 1
            continue
        if not any(f["patch"] for f in detail["files"]):
            skipped += 1     # nothing textual to classify
            continue
        append_jsonl(args.out, {
            "case_id": case_id(c), "repo": c["repo"], "kind": c["kind"], "id": c["id"],
            "date": c["date"], "title": redact(c["title"]), "author": c["author"],
            "strata": c["strata"], **detail,
        })
        written += 1
        if i % 50 == 0:
            gh.log(f"  {i}/{len(todo)}")
    gh.log(f"Wrote {written} cases to {args.out}; skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
