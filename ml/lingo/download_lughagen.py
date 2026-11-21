"""Download the LughaGen corpus files (CC BY-SA 4.0, gated on Hugging Face).

Reads a Hugging Face access token from ml/data/raw/hf_token.txt (git-ignored)
or the HF_TOKEN environment variable, downloads each language file into
ml/data/raw/lughagen/ (git-ignored), resumes partial downloads, and records
source, licence, size and SHA-256 in ml/data/raw/lughagen/provenance.json.

    python -m ml.lingo.download_lughagen
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "ml" / "data" / "raw"
OUT = RAW / "lughagen"
BASE = "https://huggingface.co/datasets/LughaGen-Organization/LughaGen-Multilingual-African-Corpus/resolve/main/data"
FILES = {
    "sheng": "sheng/Sheng_synthetic.jsonl",
    "swahili": "swahili/swahili_synthetic.jsonl",
    "luo": "luo/luo_synthetic.jsonl",
    "kikuyu": "kikuyu/kikuyu_synthetic.jsonl",
    "kamba": "kamba/kamba_synthetic.jsonl",
}


def token() -> str:
    file = RAW / "hf_token.txt"
    value = os.environ.get("HF_TOKEN") or (file.read_text(encoding="utf-8").strip() if file.is_file() else "")
    if not value:
        sys.exit("No Hugging Face token: put it in ml/data/raw/hf_token.txt or set HF_TOKEN.")
    return value


def download(url: str, target: Path, auth: str) -> None:
    have = target.stat().st_size if target.exists() else 0
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {auth}",
                                                   "User-Agent": "kenya-watch-research/1.0",
                                                   **({"Range": f"bytes={have}-"} if have else {})})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, target.open("ab" if have else "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
    except urllib.error.HTTPError as error:
        if error.code == 416:      # already complete
            return
        if error.code in (401, 403):
            sys.exit(f"{error.code} for {url}: accept the dataset's terms on its Hugging Face page with the account "
                     "that owns the token, and make sure the token has read access.")
        raise


def main() -> int:
    auth = token()
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = []
    for language, path in FILES.items():
        target = OUT / Path(path).name
        print(f"downloading {language} ...", flush=True)
        download(f"{BASE}/{path}", target, auth)
        provenance.append({
            "language": language, "source": f"{BASE}/{path}", "licence": "CC-BY-SA-4.0",
            "attribution": "LughaGen Multilingual Kenyan Language Corpus, LughaGen-Organization (JHUB Africa)",
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "downloaded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        print(f"  {target.name}: {target.stat().st_size / 1e6:.1f} MB", flush=True)
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
