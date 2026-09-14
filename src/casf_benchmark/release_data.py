"""Fetch the precomputed dashboard SQLite files from a GitHub Release.

Both dashboard DBs are larger than GitHub's 100MB hard blob limit (and Git LFS is
disabled for this repository), so they are published as release assets instead of
being committed -- see the `.gitignore` entries for them. A Weka checkout, or any
machine that rebuilt them locally, already has them on disk and never downloads.
A fresh clone (Streamlit Community Cloud, most notably) has neither, and this
module is what lets the dashboard serve the final analysis there with no Weka or
SSH access.

Which results the dashboard serves is pinned by environment, so republishing is a
new release plus an env change rather than a code change:

  CASF_DASHBOARD_RELEASE       release tag to pull from (default below)
  CASF_DASHBOARD_RELEASE_REPO  `owner/name` holding the release (default below);
                               the app itself may be served from a mirror, while
                               the assets stay on the canonical repository

Only the two files the app actually opens are fetchable. `casf_per_ligand_long.csv`
is a rebuild input for the extended analysis, never read by the dashboard, and is
deliberately not listed here.
"""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

from casf_benchmark.paths import DEFAULT_DASHBOARD_DB, DEFAULT_EXTENDED_DB

DEFAULT_RELEASE_TAG = "dashboard-data-druglike-ots-v2"
DEFAULT_RELEASE_REPO = "YerevaNN/casf-benchmark"

#: Older Streamlit Cloud secrets may still pin a pre-OTS release. Ignore them so a
#: code deploy can move Community Cloud forward without editing secrets in the UI.
LEGACY_RELEASE_TAGS = frozenset(
    {"dashboard-data-qwen-druglike", "dashboard-data-druglike-ots-v1"}
)

#: The only paths this module will ever write. Keyed by location rather than by
#: bare filename so that a DB the operator pointed us at elsewhere is never
#: silently overwritten by release contents.
RELEASE_ASSET_PATHS: tuple[Path, ...] = (DEFAULT_DASHBOARD_DB, DEFAULT_EXTENDED_DB)

CHUNK_BYTES = 1 << 20
_USER_AGENT = "casf-benchmark-dashboard"

#: Called with (bytes_downloaded, total_bytes); total is 0 when unknown.
ProgressCallback = Callable[[int, int], None]


def release_tag() -> str:
    """Release tag to fetch dashboard data from."""
    env = (os.environ.get("CASF_DASHBOARD_RELEASE") or "").strip()
    if env in LEGACY_RELEASE_TAGS:
        env = ""
    return env or DEFAULT_RELEASE_TAG


def release_pin_path() -> Path:
    return DEFAULT_DASHBOARD_DB.parent / ".dashboard_release_pin"


def read_release_pin() -> str | None:
    path = release_pin_path()
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def write_release_pin(tag: str | None = None) -> None:
    pin = tag or release_tag()
    path = release_pin_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pin}\n", encoding="utf-8")


def invalidate_stale_release_assets() -> bool:
    """Delete cached release DBs when the active tag changed.

    A missing pin does not wipe files: that is the mid-fetch state on
    Streamlit Cloud (first DB landed, second still downloading). Returns
    True when any default release asset was removed.
    """
    pin = read_release_pin()
    if pin is None or pin == release_tag():
        return False

    removed = False
    for path in RELEASE_ASSET_PATHS:
        if path.exists() and is_release_asset(path):
            path.unlink()
            removed = True
    release_pin_path().unlink(missing_ok=True)
    return removed


def mark_release_assets_current() -> None:
    """Record the active tag after any default asset lands on disk.

    The pin is written per successful fetch, not only when both DBs exist.
    Otherwise a Streamlit rerun between the two downloads treats the first
    file as stale and deletes it, then the second rename races a `.part`
    that is already gone.
    """
    if any(path.is_file() for path in RELEASE_ASSET_PATHS):
        write_release_pin()


def release_repo() -> str:
    """`owner/name` of the repository whose release holds the dashboard data."""
    return (os.environ.get("CASF_DASHBOARD_RELEASE_REPO") or "").strip() or DEFAULT_RELEASE_REPO


def asset_url(asset_name: str, tag: str | None = None, repo: str | None = None) -> str:
    """Public download URL for one asset of the pinned release."""
    return (
        f"https://github.com/{repo or release_repo()}"
        f"/releases/download/{tag or release_tag()}/{asset_name}"
    )


def _normalize(path: Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def is_release_asset(path: Path) -> bool:
    """True when `path` is one of the default DB locations published as an asset.

    Strict on purpose: a path typed into the dashboard sidebar, or set via
    `CASF_DASHBOARD_DB` / `CASF_EXTENDED_DB`, belongs to whoever set it and is
    left alone even if it happens to share a filename with a release asset.
    """
    candidate = _normalize(path)
    return any(candidate == _normalize(default) for default in RELEASE_ASSET_PATHS)


def _publish_download(src: Path, dest: Path) -> None:
    """Move a completed download into `dest`, including across filesystems."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dest)
        return
    except OSError as error:
        if error.errno != errno.EXDEV:
            # Streamlit Cloud can raise ENOENT on rename if the app reran and
            # unlinked the staging file; if dest already landed, we are done.
            if isinstance(error, FileNotFoundError) and dest.is_file():
                src.unlink(missing_ok=True)
                return
            raise
    shutil.move(str(src), str(dest))


def fetch_release_asset(
    path: Path,
    *,
    tag: str | None = None,
    repo: str | None = None,
    progress: ProgressCallback | None = None,
    timeout: float = 300.0,
) -> bool:
    """Download `path` from the pinned release; return True if a download happened.

    A no-op (returning False) when the file is already on disk or when `path` is not
    one of `RELEASE_ASSET_PATHS`, which is what keeps Weka and local rebuilds working
    untouched.

    The body is written under the system temp directory (not next to the SQLite
    in the app tree). Streamlit Cloud watches the repo and reruns the script
    when a sibling `.part` appears, which used to make `Path.replace` fail with
    ENOENT on the extended DB.
    """
    if path.exists():
        return False
    if not is_release_asset(path):
        return False

    url = asset_url(path.name, tag=tag, repo=repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                if progress is not None:
                    progress(downloaded, total)
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_download(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return True
