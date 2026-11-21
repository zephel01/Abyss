"""prebuilt.yml's reuse job: the successful prebuilt run of this commit on a release-candidate branch (ci-*), whose
archives a tag's release takes instead of building them again.

Prints run_id=<id> to $GITHUB_OUTPUT (nothing when there is no usable run: the tag's run then builds).  A ci-* run of
the same commit that is still queued or running is waited for (up to --wait minutes); a run whose two archives are
missing or expired (artifacts are kept 14 days) does not count.

  python find_prebuilt_run.py <sha> [--wait 90]
Env: GITHUB_TOKEN (actions: read), GITHUB_REPOSITORY, GITHUB_RUN_ID, GITHUB_OUTPUT.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

NAMES = ("abyss-engine-cuda-linux-x64", "abyss-engine-cuda-windows-x64")
API = "https://api.github.com"


def get(path: str) -> dict:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    req = urllib.request.Request(API + path, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def candidates(repo: str, sha: str, own: str) -> list:
    runs = get(f"/repos/{repo}/actions/workflows/prebuilt.yml/runs?head_sha={sha}&event=push&per_page=50")
    return [r for r in runs.get("workflow_runs", [])
            if r["head_branch"].startswith("ci-") and str(r["id"]) != own and r["head_sha"] == sha]


def usable(repo: str, run: dict) -> bool:
    arts = get(f"/repos/{repo}/actions/runs/{run['id']}/artifacts?per_page=100").get("artifacts", [])
    have = {a["name"] for a in arts if not a["expired"]}
    missing = [n for n in NAMES if n not in have]
    if missing:
        print(f"run {run['id']} ({run['head_branch']}): missing or expired {missing}")
    return not missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sha")
    ap.add_argument("--wait", type=float, default=90, help="minutes to wait for a ci-* run still in progress")
    a = ap.parse_args()
    repo, own = os.environ["GITHUB_REPOSITORY"], os.environ.get("GITHUB_RUN_ID", "")
    deadline = time.time() + a.wait * 60
    found = ""
    while True:
        runs = candidates(repo, a.sha, own)
        done = [r for r in runs if r["status"] == "completed" and r["conclusion"] == "success"]
        done.sort(key=lambda r: r["run_started_at"] or "", reverse=True)
        for r in done:
            if usable(repo, r):
                found = str(r["id"])
                print(f"reusing run {r['id']} ({r['head_branch']}, {r['html_url']})")
                break
        if found:
            break
        busy = [r for r in runs if r["status"] != "completed"]
        if not busy:
            print(f"no successful ci-* prebuilt run of {a.sha} with both archives: building")
            break
        if time.time() > deadline:
            print(f"ci-* run(s) {[r['id'] for r in busy]} still not done after {a.wait:g} min: building")
            break
        print(f"waiting for ci-* run(s) {[r['id'] for r in busy]} ({[r['status'] for r in busy]})", flush=True)
        time.sleep(60)
    if found and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"run_id={found}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
