#!/usr/bin/env python3
"""Read-only GitHub inventory for the HB review-routing feasibility experiment.

For each repository it records how much change history exists and how much of
it carries *observed* outcome evidence (human review, requested changes, CI
failures, reverts, fix commits). It never writes to GitHub: it issues GraphQL
queries (no mutations) and REST GETs only.

Usage:
    export GITHUB_TOKEN=...            # needs read access to the repositories
    python repo_audit.py --owner LOGIN --out audit_out
    python repo_audit.py --repos-file repos.txt --out audit_out   # owner/name per line

Re-running with the same --out resumes where it stopped, and runs for different repositories
add to the same outputs. A repository already collected is not fetched again, so to change its
--max-prs or --max-commits use a new --out.

Outputs (in --out):
    repos.csv         one row per repository
    candidates.jsonl  one row per candidate change (PR or direct commit)
    summary.md        totals and a feasibility reading
    raw/, state.json  page cache and resume cursors

The output describes private repositories. Keep it out of version control.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from common import SENSITIVE

API = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")

REVERT_RE = re.compile(r"^revert\b", re.I)
REVERTS_SHA_RE = re.compile(r"This reverts commit ([0-9a-f]{7,40})")
FIX_RE = re.compile(r"\b(fix(es|ed)?|bug|hotfix|regression|patch(es|ed)?)\b", re.I)
CI_FAILED = {"FAILURE", "ERROR"}

OWNER_REPOS_Q = """
query($login:String!,$cursor:String){
  rateLimit{remaining resetAt}
  repositoryOwner(login:$login){
    repositories(first:100, after:$cursor, ownerAffiliations:OWNER, isFork:false){
      pageInfo{hasNextPage endCursor}
      nodes{nameWithOwner}
    }
  }
}"""

PRS_Q = """
query($owner:String!,$name:String!,$cursor:String){
  rateLimit{remaining resetAt}
  repository(owner:$owner,name:$name){
    isPrivate isArchived primaryLanguage{name}
    pullRequests(first:50, after:$cursor, orderBy:{field:CREATED_AT,direction:DESC}){
      totalCount pageInfo{hasNextPage endCursor}
      nodes{
        number title createdAt mergedAt lastEditedAt state additions deletions changedFiles
        author{login __typename}
        baseRefOid headRefOid mergeCommit{oid}
        comments{totalCount}
        reviews(first:50){nodes{state author{login __typename} comments{totalCount}}}
        files(first:100){nodes{path}}
        commits(last:1){nodes{commit{statusCheckRollup{state}}}}
      }
    }
  }
}"""

COMMITS_Q = """
query($owner:String!,$name:String!,$cursor:String){
  rateLimit{remaining resetAt}
  repository(owner:$owner,name:$name){
    defaultBranchRef{name target{... on Commit{
      history(first:100, after:$cursor){
        totalCount pageInfo{hasNextPage endCursor}
        nodes{
          oid committedDate messageHeadline messageBody additions deletions
          changedFilesIfAvailable parents{totalCount}
          author{user{login __typename}}
          associatedPullRequests(first:1){nodes{number}}
          statusCheckRollup{state}
        }
      }
    }}}
  }
}"""


class RepoUnavailable(Exception):
    pass


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _sleep_until(epoch: float, why: str) -> None:
    wait = max(1.0, min(epoch - time.time() + 2, 3700))
    log(f"  {why}; sleeping {wait:.0f}s")
    time.sleep(wait)


def _request(url: str, token: str, body: dict | None = None) -> dict:
    """GET (or GraphQL POST) with retry, backoff and rate-limit handling."""
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(6):
        req = urllib.request.Request(url, data=data, headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "repo-audit-script",
        })
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                sys.exit("GitHub rejected the token (401). Check GITHUB_TOKEN.")
            if e.code == 404:
                raise RepoUnavailable(url) from e
            if e.code in (403, 429):
                retry_after = e.headers.get("Retry-After")
                reset = e.headers.get("X-RateLimit-Reset")
                if retry_after:
                    _sleep_until(time.time() + int(retry_after), "secondary rate limit")
                elif reset and e.headers.get("X-RateLimit-Remaining") == "0":
                    _sleep_until(int(reset), "rate limit exhausted")
                else:
                    _sleep_until(time.time() + 60 * (attempt + 1), f"HTTP {e.code}")
                continue
            if e.code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    sys.exit(f"Giving up on {url} after repeated failures; re-run to resume.")


def gql(token: str, query: str, variables: dict) -> dict:
    for _ in range(6):
        out = _request(f"{API}/graphql", token, {"query": query, "variables": variables})
        errors = out.get("errors") or []
        if any(e.get("type") == "RATE_LIMITED" for e in errors):
            _sleep_until(time.time() + 60, "GraphQL rate limited")
            continue
        if any(e.get("type") in ("NOT_FOUND", "FORBIDDEN") for e in errors):
            raise RepoUnavailable(errors[0].get("message", "not found"))
        if errors and not out.get("data"):
            raise RuntimeError(f"GraphQL error: {errors}")
        data = out["data"]
        rl = data.get("rateLimit") or {}
        if rl.get("remaining", 1000) < 100 and rl.get("resetAt"):
            _sleep_until(parse_ts(rl["resetAt"]).timestamp(), "GraphQL budget low")
        return data
    sys.exit("GraphQL rate limit did not clear; re-run later to resume.")


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def list_owner_repos(token: str, login: str) -> list[str]:
    repos, cursor = [], None
    while True:
        owner = gql(token, OWNER_REPOS_Q, {"login": login, "cursor": cursor})["repositoryOwner"]
        if owner is None:
            sys.exit(f"Owner {login!r} not found.")
        page = owner["repositories"]
        repos += [n["nameWithOwner"] for n in page["nodes"]]
        if not page["pageInfo"]["hasNextPage"]:
            return repos
        cursor = page["pageInfo"]["endCursor"]


# ---------- collection (resumable) ----------

class Store:
    def __init__(self, out: Path):
        self.out = out
        self.raw = out / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.state_path = out / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def path(self, repo: str, kind: str) -> Path:
        return self.raw / f"{repo.replace('/', '__')}.{kind}.jsonl"

    def append(self, repo: str, kind: str, rows: list[dict]) -> None:
        with self.path(repo, kind).open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def load(self, repo: str, kind: str, key: str) -> list[dict]:
        """Rows deduplicated by key (a crash between append and checkpoint can repeat a page)."""
        p = self.path(repo, kind)
        if not p.exists():
            return []
        seen = {}
        for line in p.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                seen[row[key]] = row
        return list(seen.values())

    def get(self, repo: str, kind: str) -> dict:
        return self.state.setdefault(repo, {}).setdefault(kind, {"cursor": None, "done": False})

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)


def collect_paged(store: Store, token: str, repo: str, kind: str, query: str, pick, limit: int = 0) -> None:
    """Page newest-first; stop after `limit` items when limit > 0."""
    st = store.get(repo, kind)
    owner, name = repo.split("/", 1)
    while not st["done"]:
        data = gql(token, query, {"owner": owner, "name": name, "cursor": st["cursor"]})
        r = data.get("repository")
        if r is None:
            raise RepoUnavailable(repo)
        if kind == "prs":
            store.state[repo]["meta"] = {
                "private": r["isPrivate"], "archived": r["isArchived"],
                "language": (r["primaryLanguage"] or {}).get("name"),
            }
        page = pick(r)
        if page is None:  # empty repository
            st["done"] = True
            st["total"] = 0
        else:
            store.append(repo, kind, page["nodes"])
            st["total"] = page["totalCount"]
            st["cursor"] = page["pageInfo"]["endCursor"]
            st["n"] = st.get("n", 0) + len(page["nodes"])
            st["done"] = not page["pageInfo"]["hasNextPage"] or bool(limit and st["n"] >= limit)
        store.save()


def collect_commit_files(store: Store, token: str, repo: str, cap: int) -> None:
    """File paths for direct (non-PR, non-merge) commits, newest first, up to cap."""
    if cap <= 0:
        return
    have = {r["oid"] for r in store.load(repo, "files", "oid")}
    direct = [c for c in store.load(repo, "commits", "oid") if is_direct(c)]
    direct.sort(key=lambda c: c["committedDate"], reverse=True)
    for c in direct[:cap]:
        if c["oid"] in have:
            continue
        try:
            d = _request(f"{API}/repos/{repo}/commits/{c['oid']}", token)
        except RepoUnavailable:
            continue
        store.append(repo, "files", [{
            "oid": c["oid"], "paths": [f["filename"] for f in d.get("files", [])],
        }])


# ---------- analysis ----------

def is_direct(c: dict) -> bool:
    return c["parents"]["totalCount"] == 1 and not c["associatedPullRequests"]["nodes"]


def is_bot(author: dict | None) -> bool:
    if not author:
        return False
    return author.get("__typename") == "Bot" or author.get("login", "").endswith("[bot]")


def strata(paths: list[str]) -> list[str]:
    return sorted(k for k, rx in SENSITIVE.items() if any(rx.search(p) for p in paths))


def analyse(store: Store, repo: str) -> tuple[dict, list[dict]]:
    prs = store.load(repo, "prs", "number")
    commits = store.load(repo, "commits", "oid")
    files = {r["oid"]: r["paths"] for r in store.load(repo, "files", "oid")}

    reverted = set()
    for c in commits:
        for sha in REVERTS_SHA_RE.findall(c.get("messageBody") or ""):
            reverted.add(sha)

    def was_reverted(oid: str | None) -> bool:
        return bool(oid) and any(oid.startswith(r) for r in reverted)

    cands = []
    for p in prs:
        author = (p["author"] or {}).get("login")
        human = [r for r in p["reviews"]["nodes"]
                 if r["author"] and not is_bot(r["author"]) and r["author"]["login"] != author]
        bots = [r for r in p["reviews"]["nodes"] if is_bot(r["author"])]
        tip = p["commits"]["nodes"]
        ci = ((tip[0]["commit"]["statusCheckRollup"] or {}).get("state")) if tip else None
        paths = [f["path"] for f in (p["files"] or {"nodes": []})["nodes"]]
        merge_oid = (p["mergeCommit"] or {}).get("oid")
        cands.append({
            "repo": repo, "kind": "pr", "id": p["number"], "title": p["title"],
            "date": p["createdAt"], "state": p["state"], "body_edited_at": p.get("lastEditedAt"),
            "base_sha": p["baseRefOid"], "head_sha": p["headRefOid"], "merge_sha": merge_oid,
            "author": author, "author_is_bot": is_bot(p["author"]),
            "additions": p["additions"], "deletions": p["deletions"],
            "changed_files": p["changedFiles"], "strata": strata(paths),
            "paths_known": bool(paths),
            "ci_state": ci,
            "observed": {
                "human_reviews": len(human),
                "human_inline_comments": sum(r["comments"]["totalCount"] for r in human),
                "changes_requested": any(r["state"] == "CHANGES_REQUESTED" for r in human),
                "bot_reviews": len(bots),
                "discussion_comments": p["comments"]["totalCount"],
                "ci_failed": ci in CI_FAILED,
                "reverted": was_reverted(merge_oid),
                "is_revert": bool(REVERT_RE.search(p["title"])),
                "is_fix": bool(FIX_RE.search(p["title"])),
            },
        })
    for c in commits:
        if not is_direct(c):
            continue
        user = (c["author"] or {}).get("user")
        ci = (c["statusCheckRollup"] or {}).get("state")
        paths = files.get(c["oid"])
        cands.append({
            "repo": repo, "kind": "commit", "id": c["oid"], "title": c["messageHeadline"],
            "date": c["committedDate"], "state": "COMMITTED",
            "base_sha": None, "head_sha": c["oid"], "merge_sha": c["oid"],
            "author": (user or {}).get("login"), "author_is_bot": is_bot(user),
            "additions": c["additions"], "deletions": c["deletions"],
            "changed_files": c["changedFilesIfAvailable"], "strata": strata(paths or []),
            "paths_known": paths is not None,
            "ci_state": ci,
            "observed": {
                "human_reviews": 0, "human_inline_comments": 0, "changes_requested": False,
                "bot_reviews": 0, "discussion_comments": 0,
                "ci_failed": ci in CI_FAILED,
                "reverted": was_reverted(c["oid"]),
                "is_revert": bool(REVERT_RE.search(c["messageHeadline"])),
                "is_fix": bool(FIX_RE.search(c["messageHeadline"])),
            },
        })
    for c in cands:
        o = c["observed"]
        # Evidence that something was wrong with, or questioned about, THIS change.
        c["has_outcome_evidence"] = bool(
            o["human_inline_comments"] or o["changes_requested"] or o["ci_failed"] or o["reverted"])

    pr_c = [c for c in cands if c["kind"] == "pr"]
    cm_c = [c for c in cands if c["kind"] == "commit"]
    human_authored = [c for c in cands if not c["author_is_bot"]]
    dates = [c["date"] for c in cands]
    meta = store.state.get(repo, {}).get("meta", {})
    strata_counts = Counter(s for c in human_authored for s in c["strata"])
    row = {
        "repo": repo,
        "private": meta.get("private"), "archived": meta.get("archived"),
        "language": meta.get("language"),
        "first_change": min(dates)[:10] if dates else "",
        "last_change": max(dates)[:10] if dates else "",
        "commits_default_branch": len(commits),
        "direct_commits": len(cm_c),
        "prs": len(pr_c),
        "prs_merged": sum(c["state"] == "MERGED" for c in pr_c),
        "prs_bot_authored": sum(c["author_is_bot"] for c in pr_c),
        "prs_human_review": sum(c["observed"]["human_reviews"] > 0 for c in pr_c),
        "prs_human_inline_comments": sum(c["observed"]["human_inline_comments"] > 0 for c in pr_c),
        "prs_changes_requested": sum(c["observed"]["changes_requested"] for c in pr_c),
        "prs_bot_review_only": sum(
            c["observed"]["bot_reviews"] > 0 and c["observed"]["human_reviews"] == 0 for c in pr_c),
        "changes_with_ci": sum(c["ci_state"] is not None for c in cands),
        "changes_ci_failed": sum(c["observed"]["ci_failed"] for c in cands),
        "reverts": sum(c["observed"]["is_revert"] for c in cands),
        "changes_reverted": sum(c["observed"]["reverted"] for c in cands),
        "fix_changes": sum(c["observed"]["is_fix"] for c in cands),
        "candidates": len(human_authored),
        "candidates_with_outcome_evidence": sum(c["has_outcome_evidence"] for c in human_authored),
        "candidates_paths_unknown": sum(not c["paths_known"] for c in human_authored),
        **{f"stratum_{k}": strata_counts.get(k, 0) for k in SENSITIVE},
    }
    return row, cands


def write_summary(out: Path, rows: list[dict], skipped: list[str]) -> None:
    def tot(k):
        return sum(r[k] or 0 for r in rows)

    n, ev = tot("candidates"), tot("candidates_with_outcome_evidence")
    lines = [
        "# Repository audit summary",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. "
        f"{len(rows)} repositories audited, {len(skipped)} skipped.",
        "",
        "| Measure | Count |",
        "|---|---|",
        f"| Candidate changes (human-authored PRs + direct commits) | {n} |",
        f"| ...of which pull requests | {tot('prs') - tot('prs_bot_authored')} |",
        f"| ...of which direct commits | {tot('direct_commits')} |",
        f"| PRs reviewed by a human other than the author | {tot('prs_human_review')} |",
        f"| PRs with human inline review comments | {tot('prs_human_inline_comments')} |",
        f"| PRs with changes requested | {tot('prs_changes_requested')} |",
        f"| PRs reviewed only by bots | {tot('prs_bot_review_only')} |",
        f"| Changes with any CI result | {tot('changes_with_ci')} |",
        f"| Changes with failed CI | {tot('changes_ci_failed')} |",
        f"| Changes later reverted (explicit revert commit) | {tot('changes_reverted')} |",
        f"| Fix-like changes (title keyword) | {tot('fix_changes')} |",
        f"| Candidates with observed outcome evidence | {ev} |",
        f"| Candidates whose file paths were not fetched | {tot('candidates_paths_unknown')} |",
        "",
        "## Path strata (heuristic, for sampling only)",
        "",
        "| Stratum | Candidates |",
        "|---|---|",
        *[f"| {k} | {tot('stratum_' + k)} |" for k in SENSITIVE],
        "",
        "## Reading",
        "",
        f"- Observed-evidence share: {ev}/{n}" + (f" ({ev / n:.1%})." if n else "."),
        "- 'Observed outcome evidence' means human inline comments, requested changes, failed CI,",
        "  or an explicit revert. Absence of evidence is not evidence the change was safe.",
        "- Fix-like changes are counted but not linked to the change that introduced the bug.",
        "- Strata are filename patterns. A change may be security-relevant without matching any.",
        "- File paths are capped at 100 per PR and are fetched for a limited number of direct",
        "  commits per repository (--commit-files).",
    ]
    if skipped:
        lines += ["", "## Skipped", "", *[f"- {s}" for s in skipped]]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--owner", help="audit every non-fork repository owned by this login")
    ap.add_argument("--repos-file", help="file with one owner/name per line")
    ap.add_argument("--repo", action="append", default=[], help="owner/name (repeatable)")
    ap.add_argument("--out", default="audit_out")
    ap.add_argument("--commit-files", type=int, default=300,
                    help="per repository, fetch file paths for up to N direct commits "
                         "(one request each; 0 disables). Default 300.")
    ap.add_argument("--max-prs", type=int, default=0,
                    help="per repository, stop after about N newest PRs (0 = all). "
                         "Use for large public repositories.")
    ap.add_argument("--max-commits", type=int, default=0,
                    help="per repository, stop after about N newest commits. 0 = the whole history, "
                         "including everything from before the repository used pull requests; "
                         "set a limit for large public repositories.")
    args = ap.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN to a token with read access to the repositories.")

    repos = list(args.repo)
    if args.repos_file:
        repos += [l.strip() for l in Path(args.repos_file).read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
    if args.owner:
        repos += list_owner_repos(token, args.owner)
    repos = list(dict.fromkeys(repos))
    if not repos:
        sys.exit("No repositories given. Use --owner, --repos-file or --repo.")

    out = Path(args.out)
    store = Store(out)
    rows, skipped, all_cands = [], [], []
    for i, repo in enumerate(repos, 1):
        log(f"[{i}/{len(repos)}] {repo}")
        try:
            collect_paged(store, token, repo, "prs", PRS_Q, lambda r: r["pullRequests"], args.max_prs)
            collect_paged(
                store, token, repo, "commits", COMMITS_Q,
                lambda r: r["defaultBranchRef"]["target"]["history"] if r["defaultBranchRef"] else None,
                args.max_commits)
            collect_commit_files(store, token, repo, args.commit_files)
        except RepoUnavailable as e:
            log(f"  skipped: {e}")
            skipped.append(repo)
            continue
        row, cands = analyse(store, repo)
        rows.append(row)
        all_cands += cands

    # Repositories audited into this --out by earlier runs stay in the outputs. Without this a
    # second run (for example one adding public repositories) would replace the candidate list.
    earlier = [r for r, st in store.state.items() if r not in repos
               and st.get("prs", {}).get("done") and st.get("commits", {}).get("done")]
    for repo in sorted(earlier):
        row, cands = analyse(store, repo)
        rows.append(row)
        all_cands += cands
    if earlier:
        log(f"Kept {len(earlier)} repositories from earlier runs: {', '.join(sorted(earlier))}")
    rows.sort(key=lambda r: r["repo"])

    if rows:
        with (out / "repos.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with (out / "candidates.jsonl").open("w") as f:
        for c in sorted(all_cands, key=lambda c: (c["repo"], c["date"])):
            f.write(json.dumps(c) + "\n")
    write_summary(out, rows, skipped)
    log(f"Wrote {out}/summary.md, repos.csv, candidates.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
