#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


def download_qwen3_1_7b(
    repo_id: str = "Qwen/Qwen3-1.7B", target_subdir: str = "Qwen3-1-7B"
) -> Path:
    base_dir = Path(__file__).resolve().parent
    target_dir = base_dir / target_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )
    return target_dir


def main() -> None:
    download_qwen3_1_7b()


if __name__ == "__main__":
    main()
