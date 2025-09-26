#!/usr/bin/env python3
"""
Create GitHub issues from Markdown files in .github/ISSUE_TEMPLATE.

- Parses simple YAML-like frontmatter for fields: name (title), labels, about.
- Uses the remainder of the file as the issue body (optionally prefixed by about).
- Requires a GitHub token (env GITHUB_TOKEN or --token).
- Auto-detects repo from git remote origin unless --repo is provided.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.request import Request, urlopen


FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def echo(msg: str) -> None:
    # Console-friendly logs (no emojis/unicode per Windows constraints)
    print(msg)


def detect_repo_slug() -> Optional[str]:
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            text=True,
        ).strip()
    except Exception:
        return None

    if not url:
        return None

    # Handle both HTTPS and SSH remotes
    # Examples:
    #  - https://github.com/InfantLab/VideoAnnotator.git
    #  - git@github.com:InfantLab/VideoAnnotator.git
    slug = None
    if url.startswith("http"):
        m = re.search(r"github\.com[:/]{1,2}([^/]+)/([^/.]+)", url)
        if m:
            slug = f"{m.group(1)}/{m.group(2)}"
    elif url.startswith("git@"):
        m = re.search(r"github\.com:(.+?)/([^/.]+)", url)
        if m:
            slug = f"{m.group(1)}/{m.group(2)}"
    return slug


def parse_frontmatter_and_body(text: str) -> Tuple[Dict[str, str], str]:
    """Return (frontmatter_dict, body_text). Accepts simple key: value pairs.

    labels may be provided as a comma-separated string.
    """
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    header, body = m.group(1), m.group(2)
    fm: Dict[str, str] = {}
    for line in header.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        fm[key.strip()] = val.strip().strip('"')
    return fm, body.lstrip()


def ensure_labels(repo: str, token: str, labels: List[str]) -> None:
    # Best-effort: create labels that don't exist.
    # Use GET /labels to fetch existing, then create missing with POST /labels.
    # If lacking perms, issue creation will proceed and GitHub may auto-create or ignore.
    if not labels:
        return
    api = f"https://api.github.com/repos/{repo}/labels?per_page=100"
    req = Request(api, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    try:
        with urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            existing = {lbl.get("name", "") for lbl in data}
    except Exception:
        return  # Non-fatal; continue

    for name in labels:
        if name in existing or not name:
            continue
        payload = json.dumps({"name": name}).encode("utf-8")
        create_req = Request(
            f"https://api.github.com/repos/{repo}/labels",
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(create_req) as _:
                pass
        except Exception:
            # Ignore; not critical
            continue


def create_issue(repo: str, token: str, title: str, body: str, labels: List[str]) -> Tuple[int, str]:
    payload = {
        "title": title,
        "body": body,
        "labels": labels,
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(req) as resp:
        resp_data = json.loads(resp.read().decode("utf-8"))
    return resp_data.get("number"), resp_data.get("html_url")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Create GitHub issues from Markdown templates")
    p.add_argument("--repo", help="owner/repo (auto-detected from git remote if omitted)")
    p.add_argument("--token", help="GitHub token (or set GITHUB_TOKEN)")
    p.add_argument("--dir", default=str(Path(".github/ISSUE_TEMPLATE")), help="Directory containing .md files")
    p.add_argument("--create", action="store_true", help="Actually create issues (otherwise dry-run)")
    p.add_argument("--remove-after", action="store_true", help="Remove files after successful creation")
    args = p.parse_args(argv)

    repo = args.repo or detect_repo_slug()
    if not repo:
        echo("[ERROR] Could not detect repo slug; pass --repo owner/repo")
        return 2

    token = args.token or os.getenv("GITHUB_TOKEN")
    if args.create and not token:
        echo("[ERROR] Missing token. Set GITHUB_TOKEN or pass --token.")
        return 2

    base = Path(args.dir)
    if not base.exists():
        echo(f"[ERROR] Directory not found: {base}")
        return 2

    md_files = sorted([p for p in base.glob("*.md")])
    if not md_files:
        echo(f"[WARNING] No .md files found in {base}")
        return 0

    echo(f"[START] Repo: {repo}")
    echo(f"[START] Mode: {'CREATE' if args.create else 'DRY-RUN'}")

    created: List[Tuple[Path, int, str]] = []

    for f in md_files:
        text = f.read_text(encoding="utf-8")
        fm, body = parse_frontmatter_and_body(text)
        title = fm.get("name") or f.stem.replace("-", " ").title()
        labels_raw = fm.get("labels", "").strip()
        labels = [s.strip() for s in labels_raw.split(",") if s.strip()] if labels_raw else []
        about = fm.get("about", "").strip()
        if about:
            body = f"_{about}_\n\n" + body

        echo(f"[START] Issue from: {f.name}")
        echo(f"  Title: {title}")
        if labels:
            echo(f"  Labels: {', '.join(labels)}")

        if args.create:
            try:
                ensure_labels(repo, token, labels)
                num, url = create_issue(repo, token, title, body, labels)
                echo(f"[OK] Created issue #{num}: {url}")
                created.append((f, num, url))
            except HTTPError as e:
                try:
                    err_body = e.read().decode("utf-8")
                except Exception:
                    err_body = str(e)
                echo(f"[ERROR] HTTP {e.code} creating issue for {f.name}: {err_body}")
                return 1
            except Exception as e:
                echo(f"[ERROR] Failed to create issue for {f.name}: {e}")
                return 1
        else:
            echo("[OK] Dry-run only. No issue created.")

    if args.create and args.remove_after:
        for f, _, _ in created:
            try:
                f.unlink()
                echo(f"[OK] Removed: {f}")
            except Exception as e:
                echo(f"[WARNING] Could not remove {f}: {e}")

    echo("[OK] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
