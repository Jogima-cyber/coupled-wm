#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Download the released weights from the Hugging Face Hub into ``weights/``.

    python scripts/download_weights.py --repo-id Jogima-cyber/coupled-global-local-wm-rl --out weights/

The repository holds three pretrained dirs (``config.json`` + ``model.safetensors``):
``tokenizer/`` and ``dynamics/`` (the global world model) and ``rssm/`` (the RL warm start of
the local world model). ``configs/rl/g1_dmo.yaml`` reads them from ``${WEIGHTS_DIR:-weights}``.
After the download each dir is checked to be a complete pretrained dir and the dynamics model's
recorded tokenizer sha256 is compared with the downloaded tokenizer.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

try:
    import coupled_wm  # noqa: F401
except ModuleNotFoundError:  # source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupled_wm.common.checkpoint import is_pretrained_dir, load_pretrained_config
from coupled_wm.common.logging import setup_logging
from coupled_wm.world_model.loading import check_pairing

log = logging.getLogger("download_weights")

COMPONENTS = ("tokenizer", "dynamics", "rssm")


def verify(out_dir: str | Path, components: Sequence[str] = COMPONENTS, check_sha: bool = True) -> None:
    """Raise unless every component is a pretrained dir (and the tokenizer/dynamics pair matches)."""
    out_dir = Path(out_dir)
    configs = {}
    for name in components:
        if not is_pretrained_dir(out_dir / name):
            raise FileNotFoundError(f"{out_dir / name} is not a pretrained dir (config.json + model.safetensors)")
        configs[name] = load_pretrained_config(out_dir / name)
    if "tokenizer" in configs and "dynamics" in configs:
        check_pairing(configs["tokenizer"], configs["dynamics"], out_dir / "tokenizer" if check_sha else None)
    log.info("verified %s in %s", ", ".join(components), out_dir)


def download(repo_id: str, out_dir: str | Path, components: Sequence[str] = COMPONENTS, revision: str | None = None,
             token: str | None = None, check_sha: bool = True) -> Path:
    """Download ``components`` of ``repo_id`` into ``out_dir`` and verify them."""
    from huggingface_hub import snapshot_download

    unknown = sorted(set(components) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"unknown components {unknown}; choose from {list(COMPONENTS)}")
    out_dir = Path(out_dir)
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(out_dir), token=token,
                      allow_patterns=[f"{name}/*" for name in components])
    verify(out_dir, components, check_sha=check_sha)
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the released pretrained weights from the Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repository, <org>/<name>")
    parser.add_argument("--out", default="weights", help="output dir (default: weights/)")
    parser.add_argument("--components", nargs="+", default=list(COMPONENTS), choices=COMPONENTS,
                        help="which pretrained dirs to download (default: all)")
    parser.add_argument("--revision", help="branch, tag or commit of the repository")
    parser.add_argument("--token", help="Hugging Face token (default: the cached login, if any)")
    parser.add_argument("--no-sha-check", action="store_true",
                        help="skip hashing the tokenizer weights for the tokenizer/dynamics pairing check")
    args = parser.parse_args(argv)
    setup_logging()
    out = download(args.repo_id, args.out, args.components, revision=args.revision, token=args.token,
                   check_sha=not args.no_sha_check)
    log.info("weights are in %s", out)


if __name__ == "__main__":
    main()
