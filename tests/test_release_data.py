"""Tests for fetching dashboard data from a GitHub Release.

No network: the transport is faked so the guards -- which of the two files may be
written, when a fetch is skipped, and what is left behind on failure -- are what
gets tested.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from casf_benchmark import release_data
from casf_benchmark.paths import DEFAULT_DASHBOARD_DB, DEFAULT_EXTENDED_DB


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, content_length: bool = True) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))} if content_length else {}

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def fake_urlopen(payload: bytes, *, content_length: bool = True, requested: list[str] | None = None):
    def opener(request, timeout=None):  # noqa: ANN001 - mirrors urllib's signature
        if requested is not None:
            requested.append(request.full_url)
        return FakeResponse(payload, content_length=content_length)

    return opener


def test_release_tag_and_repo_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CASF_DASHBOARD_RELEASE", raising=False)
    monkeypatch.delenv("CASF_DASHBOARD_RELEASE_REPO", raising=False)
    assert release_data.release_tag() == release_data.DEFAULT_RELEASE_TAG
    assert release_data.release_repo() == release_data.DEFAULT_RELEASE_REPO


def test_release_tag_is_env_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    # Republishing results must not need a code change.
    monkeypatch.setenv("CASF_DASHBOARD_RELEASE", "dashboard-data-2027")
    monkeypatch.setenv("CASF_DASHBOARD_RELEASE_REPO", "MenuaB/casf-benchmark")
    assert release_data.release_tag() == "dashboard-data-2027"
    assert release_data.release_repo() == "MenuaB/casf-benchmark"
    assert release_data.asset_url("x.sqlite") == (
        "https://github.com/MenuaB/casf-benchmark/releases/download/dashboard-data-2027/x.sqlite"
    )


def test_blank_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CASF_DASHBOARD_RELEASE", "   ")
    assert release_data.release_tag() == release_data.DEFAULT_RELEASE_TAG


def test_legacy_release_tag_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CASF_DASHBOARD_RELEASE", "dashboard-data-qwen-druglike")
    assert release_data.release_tag() == release_data.DEFAULT_RELEASE_TAG


def test_a_formerly_pinned_default_now_falls_back_to_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    # dashboard-data-druglike-ots-v1 used to be DEFAULT_RELEASE_TAG itself; a secret
    # still holding it should move forward to "latest" rather than stay frozen.
    monkeypatch.setenv("CASF_DASHBOARD_RELEASE", "dashboard-data-druglike-ots-v1")
    assert release_data.release_tag() == release_data.LATEST_TAG


def test_default_release_tag_is_latest() -> None:
    assert release_data.DEFAULT_RELEASE_TAG == release_data.LATEST_TAG


def test_asset_url_for_latest_uses_the_latest_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CASF_DASHBOARD_RELEASE", raising=False)
    assert release_data.asset_url("x.sqlite", repo="o/r") == (
        "https://github.com/o/r/releases/latest/download/x.sqlite"
    )


def test_asset_url_for_a_pinned_tag_uses_the_tagged_download(monkeypatch: pytest.MonkeyPatch) -> None:
    assert release_data.asset_url("x.sqlite", tag="dashboard-data-2027", repo="o/r") == (
        "https://github.com/o/r/releases/download/dashboard-data-2027/x.sqlite"
    )
def test_invalidate_without_pin_keeps_existing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dashboard = tmp_path / "casf_analysis_dashboard.sqlite"
    extended = tmp_path / "extended_casf_analysis.sqlite"
    dashboard.write_bytes(b"first of two")
    monkeypatch.setattr(release_data, "RELEASE_ASSET_PATHS", (dashboard, extended))
    monkeypatch.setattr(release_data, "DEFAULT_DASHBOARD_DB", dashboard)
    monkeypatch.setattr(release_data, "release_pin_path", lambda: tmp_path / ".dashboard_release_pin")

    assert release_data.invalidate_stale_release_assets() is False
    assert dashboard.read_bytes() == b"first of two"


def test_mark_release_assets_current_pins_after_the_first_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dashboard = tmp_path / "casf_analysis_dashboard.sqlite"
    extended = tmp_path / "extended_casf_analysis.sqlite"
    dashboard.write_bytes(b"ok")
    pin = tmp_path / ".dashboard_release_pin"
    monkeypatch.setattr(release_data, "RELEASE_ASSET_PATHS", (dashboard, extended))
    monkeypatch.setattr(release_data, "release_pin_path", lambda: pin)

    release_data.mark_release_assets_current()
    assert pin.read_text(encoding="utf-8").strip() == release_data.DEFAULT_RELEASE_TAG


def test_invalidate_stale_release_assets_deletes_cached_dbs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dashboard = tmp_path / "casf_analysis_dashboard.sqlite"
    extended = tmp_path / "extended_casf_analysis.sqlite"
    dashboard.write_bytes(b"old")
    extended.write_bytes(b"old")
    pin = tmp_path / ".dashboard_release_pin"
    pin.write_text("dashboard-data-qwen-druglike\n", encoding="utf-8")
    monkeypatch.setattr(release_data, "RELEASE_ASSET_PATHS", (dashboard, extended))
    monkeypatch.setattr(release_data, "DEFAULT_DASHBOARD_DB", dashboard)
    monkeypatch.setattr(release_data, "release_pin_path", lambda: pin)

    assert release_data.invalidate_stale_release_assets() is True
    assert not dashboard.exists()
    assert not extended.exists()
    assert not pin.exists()


def test_only_the_two_default_db_paths_are_release_assets() -> None:
    assert release_data.is_release_asset(DEFAULT_DASHBOARD_DB)
    assert release_data.is_release_asset(DEFAULT_EXTENDED_DB)
    # casf_per_ligand_long.csv is a rebuild input the dashboard never opens.
    assert not release_data.is_release_asset(DEFAULT_DASHBOARD_DB.parent / "casf_per_ligand_long.csv")


def test_a_same_named_file_elsewhere_is_not_an_asset(tmp_path: Path) -> None:
    # A DB the operator pointed us at is theirs; sharing a filename is not consent.
    assert not release_data.is_release_asset(tmp_path / DEFAULT_DASHBOARD_DB.name)


def test_fetch_skips_a_file_already_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # This is what keeps Weka and local rebuilds working untouched.
    target = tmp_path / "casf_analysis_dashboard.sqlite"
    target.write_bytes(b"local rebuild")
    monkeypatch.setattr(release_data, "RELEASE_ASSET_PATHS", (target,))
    monkeypatch.setattr(release_data.urllib.request, "urlopen", fake_urlopen(b"remote"))

    assert release_data.fetch_release_asset(target) is False
    assert target.read_bytes() == b"local rebuild"


def test_fetch_refuses_a_non_default_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "casf_analysis_dashboard.sqlite"
    monkeypatch.setattr(release_data, "RELEASE_ASSET_PATHS", (tmp_path / "other.sqlite",))
    monkeypatch.setattr(release_data.urllib.request, "urlopen", fake_urlopen(b"remote"))

    assert release_data.fetch_release_asset(target) is False
    assert not target.exists()


def test_fetch_downloads_and_reports_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"x" * (release_data.CHUNK_BYTES * 2 + 7)
    target = tmp_path / "extended_casf_analysis.sqlite"
    requested: list[str] = []
    monkeypatch.setattr(release_data, "RELEASE_ASSET_PATHS", (target,))
    monkeypatch.setattr(
        release_data.urllib.request, "urlopen", fake_urlopen(payload, requested=requested)
    )

    seen: list[tuple[int, int]] = []
    record = lambda done, total: seen.append((done, total))  # noqa: E731
    assert release_data.fetch_release_asset(target, tag="t", repo="o/r", progress=record) is True
    assert target.read_bytes() == payload
    assert requested == ["https://github.com/o/r/releases/download/t/extended_casf_analysis.sqlite"]
    assert seen[0] == (0, len(payload))
    assert seen[-1] == (len(payload), len(payload))
    assert not list(tmp_path.glob("*.part"))


def test_fetch_tolerates_a_missing_content_length(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "extended_casf_analysis.sqlite"
    monkeypatch.setattr(release_data, "RELEASE_ASSET_PATHS", (target,))
    monkeypatch.setattr(
        release_data.urllib.request, "urlopen", fake_urlopen(b"payload", content_length=False)
    )

    seen: list[tuple[int, int]] = []
    record = lambda done, total: seen.append((done, total))  # noqa: E731
    assert release_data.fetch_release_asset(target, progress=record) is True
    assert target.read_bytes() == b"payload"
    assert all(total == 0 for _, total in seen)


def test_a_failed_fetch_leaves_no_truncated_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A half-written DB must never be left where a later run would trust it."""
    target = tmp_path / "casf_analysis_dashboard.sqlite"
    monkeypatch.setattr(release_data, "RELEASE_ASSET_PATHS", (target,))

    def failing_urlopen(request, timeout=None):  # noqa: ANN001 - mirrors urllib's signature
        class Dying(FakeResponse):
            def read(self, size: int | None = -1) -> bytes:
                raise ConnectionResetError("dropped mid-download")

        return Dying(b"never arrives")

    monkeypatch.setattr(release_data.urllib.request, "urlopen", failing_urlopen)

    with pytest.raises(ConnectionResetError):
        release_data.fetch_release_asset(target)
    assert not target.exists()
    assert not list(tmp_path.glob("*.part"))
