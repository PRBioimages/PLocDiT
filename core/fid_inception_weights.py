"""Predownload the Inception weights used by pytorch-fid (with CN mirrors).

pytorch_fid pulls the file from GitHub releases, which is slow from mainland
China. Call ``ensure_fid_inception_weights()`` before instantiating
``InceptionV3(use_fid_inception=True)``: the weights are placed at
``$TORCH_HOME/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth`` so the
later ``load_state_dict_from_url`` hits the cache instead of the network.

Environment variables (optional):
  FID_INCEPTION_MIRROR   ghfast | ghproxy | ghproxy_cc | direct | auto (default auto)
  FID_INCEPTION_URL      full download URL (takes priority over mirror)
  FID_INCEPTION_WEIGHTS  path to an existing local .pth (copied into the cache dir)
"""
from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

FID_WEIGHTS_FILENAME = "pt_inception-2015-12-05-6726825d.pth"
FID_WEIGHTS_URL_GITHUB = (
    "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/"
    f"{FID_WEIGHTS_FILENAME}"
)

# Prefix mirrors: mirror_url = prefix + original GitHub URL
MIRROR_PREFIXES: dict[str, str] = {
    "ghfast": "https://ghfast.top/",
    "ghproxy": "https://mirror.ghproxy.com/",
    "ghproxy_cc": "https://ghproxy.cc/",
}

# Auto mode tries mirrors first, then the direct URL
_AUTO_MIRROR_ORDER = ("ghfast", "ghproxy", "ghproxy_cc")


def default_checkpoints_dir() -> Path:
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints"


def _dest_path(cache_dir: Optional[Path] = None) -> Path:
    return (Path(cache_dir) if cache_dir is not None else default_checkpoints_dir()) / FID_WEIGHTS_FILENAME


def _is_valid_weights_file(path: Path, min_bytes: int = 10_000_000) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def build_download_urls(
    mirror: str = "auto",
    custom_url: Optional[str] = None,
) -> list[str]:
    if custom_url:
        return [custom_url.strip()]
    m = (mirror or "auto").strip().lower()
    if m in ("", "auto"):
        urls = [MIRROR_PREFIXES[k] + FID_WEIGHTS_URL_GITHUB for k in _AUTO_MIRROR_ORDER]
        urls.append(FID_WEIGHTS_URL_GITHUB)
        return urls
    if m in ("none", "direct", "github"):
        return [FID_WEIGHTS_URL_GITHUB]
    prefix = MIRROR_PREFIXES.get(m)
    if prefix is None:
        raise ValueError(
            f"Unknown mirror={mirror!r}, options: auto, direct, "
            + ", ".join(sorted(MIRROR_PREFIXES))
        )
    return [prefix + FID_WEIGHTS_URL_GITHUB]


def _download_one(url: str, dest: Path, verbose: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if not verbose or total_size <= 0:
            return
        done = block_num * block_size
        pct = min(100.0, 100.0 * done / total_size)
        mb = 1e-6
        print(
            f"\r[fid-weights] {pct:5.1f}%  {done * mb:.1f}/{total_size * mb:.1f} MB",
            end="",
            flush=True,
        )

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)
        if verbose:
            print()
        tmp.replace(dest)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise


def ensure_fid_inception_weights(
    cache_dir: Optional[os.PathLike] = None,
    weights_path: Optional[str] = None,
    url: Optional[str] = None,
    mirror: Optional[str] = None,
    verbose: bool = True,
) -> Path:
    """Ensure the FID Inception weights exist in the torch hub checkpoints dir.

    Returns:
        Absolute path to the weights file.
    """
    dest = _dest_path(Path(cache_dir) if cache_dir is not None else None)

    local = (weights_path or os.environ.get("FID_INCEPTION_WEIGHTS", "") or "").strip()
    if local:
        src = Path(local).expanduser().resolve()
        if not _is_valid_weights_file(src, min_bytes=1_000_000):
            raise FileNotFoundError(f"Local FID Inception weights invalid or too small: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        if verbose:
            print(f"[fid-weights] copied local weights -> {dest}")
        return dest

    if _is_valid_weights_file(dest):
        if verbose:
            print(f"[fid-weights] cache hit, skipping download: {dest}")
        return dest

    env_url = (os.environ.get("FID_INCEPTION_URL", "") or "").strip()
    env_mirror = (os.environ.get("FID_INCEPTION_MIRROR", "") or "").strip()
    mirror_eff = (mirror if mirror is not None else env_mirror or "auto").strip().lower()
    url_eff = (url or env_url or None)

    urls = build_download_urls(mirror=mirror_eff, custom_url=url_eff)
    last_err: Optional[BaseException] = None
    for u in urls:
        try:
            if verbose:
                print(f"[fid-weights] downloading: {u}")
            _download_one(u, dest, verbose=verbose)
            if not _is_valid_weights_file(dest):
                dest.unlink(missing_ok=True)
                raise RuntimeError("download finished but file size is invalid")
            if verbose:
                print(f"[fid-weights] done -> {dest}")
            return dest
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            last_err = e
            if verbose:
                print(f"[fid-weights] failed ({e!r}), trying next source...")
    raise RuntimeError(
        "All FID Inception weight sources failed. Download manually and pass "
        "--fid-inception-weights /path/to/"
        + FID_WEIGHTS_FILENAME
        + " or export FID_INCEPTION_WEIGHTS=...;"
        " alternatively export FID_INCEPTION_URL=<mirror direct link>"
    ) from last_err
