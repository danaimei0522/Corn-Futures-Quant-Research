#!/usr/bin/env python3
"""Check that the public candidate remains code-only and hash-consistent."""
from pathlib import Path
import hashlib, json, re, sys

ROOT = Path(__file__).resolve().parents[1]
manifest_path = ROOT / "release_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

def fail(msg):
    print("FAIL:", msg)
    raise SystemExit(1)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = [p for p in ROOT.rglob("*") if p.is_file() and p.name != "release_manifest.json"]
listed = set(manifest.get("files", {}))
actual = {p.relative_to(ROOT).as_posix() for p in files}
if actual != listed:
    fail(f"manifest inventory mismatch: unlisted={sorted(actual-listed)}, missing={sorted(listed-actual)}")
for rel, expected in manifest["files"].items():
    if sha(ROOT / rel) != expected:
        fail(f"SHA-256 mismatch: {rel}")

forbidden_ext = {".csv", ".bin", ".png", ".jpg", ".jpeg", ".pdf", ".parquet", ".pkl", ".pickle", ".db", ".sqlite", ".sqlite3", ".xlsx"}
for p in files:
    if p.suffix.lower() in forbidden_ext:
        fail(f"data/result or visual artifact is not allowed in code-only candidate: {p.relative_to(ROOT)}")
    if p.relative_to(ROOT).parts[0] in {"results", "figures", "data", "output"}:
        fail(f"forbidden public content directory: {p.relative_to(ROOT)}")

secret_patterns = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})\b"),
    re.compile(r"(?i)(?:api[_-]?key|password|passwd|secret|access[_-]?token)\s*[=:]\s*['\"][^'\"]{8,}['\"]"),
    re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
]
for p in files:
    if p.suffix.lower() not in {".md", ".py", ".json", ".txt"}: continue
    text = p.read_text(encoding="utf-8")
    if re.search(r"/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+", text):
        fail(f"machine-specific absolute path: {p.relative_to(ROOT)}")
    for pat in secret_patterns:
        if pat.search(text): fail(f"possible credential pattern: {p.relative_to(ROOT)}")

pyfiles = list(ROOT.rglob("*.py"))
for p in pyfiles:
    try: compile(p.read_text(encoding="utf-8"), str(p), "exec")
    except SyntaxError as e: fail(f"Python syntax error: {p.relative_to(ROOT)}: {e}")

readme=(ROOT/"README.md").read_text(encoding="utf-8")
if "READY TO PUBLISH AS CODE-ONLY" not in readme or "No market data" not in readme:
    fail("README must state code-only scope and publication status")
print(f"PASS: {len(files)} files hash-verified; {len(pyfiles)} Python scripts compile; no data/results/figures or common secret patterns found.")
