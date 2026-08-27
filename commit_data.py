#!/usr/bin/env python3
"""Commit generated `data/` files through the GitHub API, so they land signed.

A `git push` from Actions produces an UNVERIFIED commit. The runner holds no
signing key and `github-actions[bot]` is only an author email, so GitHub has
nothing to verify -- every ledger commit this repo has ever made shows that
way. Commits created through the GraphQL `createCommitOnBranch` mutation are
signed by GitHub itself with the same content and the same bot author, and
show as Verified.

Two properties of the old `git add` / `git commit` / `git pull --rebase` /
`git push` sequence are load-bearing and are preserved here:

* **A failure must never cost a slate.** The build commits pregame snapshots
  and `grade_leans.ingest()` only accepts a row whose snapshot predates first
  pitch, so rows that fail to land cannot be re-derived later without
  lookahead. The API path is newer and less proven than `git push`, so any
  failure of it falls back to exactly the old sequence: the data lands
  unverified rather than not at all, with a workflow warning saying so.
* **The push must not retrigger the workflow.** Commits made with the
  repository's `GITHUB_TOKEN` do not create new workflow runs, and that holds
  for API-created commits as much as for pushed ones.

Concurrency is handled the way `git push` handles it, not by blind retry:
`expectedHeadOid` pins the branch tip the commit is built on, and if the tip
moved under us we re-read it and retry ONLY when the intervening commit
touched none of the files we are sending. If it touched one of ours, this
exits non-zero rather than silently overwriting someone else's write -- the
same refusal `git pull --rebase` gives on a conflict.

Usage:
    python commit_data.py --branch main --message "ledger 2026-08-27T13:18Z"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time

import requests

API_URL = "https://api.github.com/graphql"

MUTATION = """
mutation ($input: CreateCommitOnBranchInput!) {
  createCommitOnBranch(input: $input) {
    commit { oid url }
  }
}
"""


def log(message):
    print(message, flush=True)


def git(*args, check=True):
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def changed_paths(paths, status_text=None):
    """Split the worktree's pending changes under `paths` into (adds, dels).

    Reads `git status --porcelain -z`, which reports untracked files but never
    ignored ones -- so an atomic-write temp file left behind by an interrupted
    writer is excluded here for the same reason `.gitignore` excludes it from
    `git add`. Classification is by what is on disk now rather than by the
    status code, so a rename arrives as an addition plus a deletion without
    this function having to know the rename spelling.
    """
    if status_text is None:
        status_text = git(
            "status", "--porcelain", "-z", "--untracked-files=all", "--", *paths
        )
    fields = [f for f in status_text.split("\0") if f]
    seen, adds, dels = set(), [], []
    for field in fields:
        # "XY path"; rename/copy entries emit the old path as its own field,
        # which carries no status prefix and is picked up by the else branch.
        path = field[3:] if len(field) > 3 and field[2] == " " else field
        if path in seen:
            continue
        seen.add(path)
        (adds if os.path.exists(path) else dels).append(path)
    return sorted(adds), sorted(dels)


def file_changes(adds, dels):
    """Build the mutation's fileChanges payload (contents are base64)."""
    additions = []
    for path in adds:
        with open(path, "rb") as handle:
            contents = base64.b64encode(handle.read()).decode("ascii")
        additions.append({"path": path, "contents": contents})
    return {
        "additions": additions,
        "deletions": [{"path": path} for path in dels],
    }


def payload_bytes(changes):
    return sum(len(a["contents"]) for a in changes["additions"])


def graphql(token, query, variables, timeout=60):
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        data=json.dumps({"query": query, "variables": variables}),
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("errors"):
        raise RuntimeError(json.dumps(body["errors"]))
    return body["data"]


def remote_head(branch):
    """The branch tip as the remote sees it right now."""
    out = git("ls-remote", "origin", f"refs/heads/{branch}").strip()
    if not out:
        raise RuntimeError(f"origin has no branch {branch}")
    return out.split()[0]


def touches_ours(old_oid, new_oid, ours):
    """Did the commit(s) between two tips write any file we are sending?"""
    git("fetch", "--quiet", "origin", new_oid, check=False)
    names = git("diff", "--name-only", old_oid, new_oid, check=False).split()
    return sorted(set(names) & set(ours))


def api_commit(token, repo, branch, message, changes, attempts=3):
    for attempt in range(1, attempts + 1):
        head = remote_head(branch)
        variables = {"input": {
            "branch": {
                "repositoryNameWithOwner": repo,
                "branchName": branch,
            },
            "message": {"headline": message},
            "expectedHeadOid": head,
            "fileChanges": changes,
        }}
        try:
            data = graphql(token, MUTATION, variables)
        except RuntimeError as exc:
            moved = remote_head(branch)
            if moved == head or attempt == attempts:
                raise
            ours = [a["path"] for a in changes["additions"]]
            ours += [d["path"] for d in changes["deletions"]]
            clash = touches_ours(head, moved, ours)
            if clash:
                raise RuntimeError(
                    f"{branch} moved from {head[:7]} to {moved[:7]} and the new "
                    f"commit wrote {', '.join(clash)} -- refusing to overwrite"
                ) from exc
            log(f"  {branch} moved to {moved[:7]} beneath us; retrying")
            time.sleep(2 ** attempt)
            continue
        return data["createCommitOnBranch"]["commit"]
    raise RuntimeError("unreachable")


def git_commit_and_push(branch, message):
    """The pre-API sequence, kept verbatim as the fallback path."""
    git("config", "user.name", "github-actions[bot]")
    git("config", "user.email",
        "github-actions[bot]@users.noreply.github.com")
    git("add", "data/")
    nothing_staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], check=False
    ).returncode == 0
    if not nothing_staged:
        git("commit", "-m", message)
    git("pull", "--rebase", "origin", branch)
    # The rebase is the one step that can introduce conflict-marker text into
    # a CSV, which is why the old sequence validated a second time here. The
    # API path has no rebase and validates once; this path keeps both.
    subprocess.run([sys.executable, "validate_data_files.py"], check=True)
    git("push", "origin", f"HEAD:{branch}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--path", action="append", default=None,
                        help="paths to commit (default: data)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="fail instead of falling back to git push")
    args = parser.parse_args(argv)
    paths = args.path or ["data"]

    adds, dels = changed_paths(paths)
    if not adds and not dels:
        log("commit: no data changes")
        return 0

    changes = file_changes(adds, dels)
    log(f"commit: {len(adds)} written, {len(dels)} removed, "
        f"{payload_bytes(changes) // 1024} KiB encoded")
    for path in adds + dels:
        log(f"  {path}")

    token = os.environ.get("GITHUB_TOKEN") or ""
    repo = os.environ.get("GITHUB_REPOSITORY") or ""
    try:
        if not token or not repo:
            raise RuntimeError("GITHUB_TOKEN / GITHUB_REPOSITORY not set")
        commit = api_commit(token, repo, args.branch, args.message, changes)
    except Exception as exc:  # noqa: BLE001
        if args.no_fallback:
            raise
        # Losing a pregame slate is irreversible; an unverified commit is not.
        log(f"::warning::signed commit failed ({exc}); "
            "falling back to an unverified git push")
        git_commit_and_push(args.branch, args.message)
        return 0
    log(f"commit: signed {commit['oid'][:7]} {commit['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
