from __future__ import annotations

import importlib.util
from types import ModuleType

import pandas as pd

from casf_benchmark.paths import REPO_ROOT


def load_dashboard() -> ModuleType:
    path = REPO_ROOT / "apps" / "dashboard" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("dashboard_public_methods", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_qwen_allowlist_preserves_non_qwen_rows() -> None:
    dashboard = load_dashboard()
    frame = pd.DataFrame(
        [
            {
                "family": "rdkit_random",
                "display_label": "RDKit random",
                "method": "rdkit_random_fixed",
            },
            {
                "family": "qwen_0p6b_bigdata",
                "display_label": "old label",
                "method": "qwen_0p6b_bigdata_fixed",
            },
            {
                "family": "qwen_1p7b_4e_step29600",
                "display_label": "Qwen 1.7B 4e (step 29600)",
                "method": "qwen_1p7b_4e_step29600_fixed",
            },
        ]
    )

    result = dashboard.filter_public_methods(frame)

    assert result["family"].tolist() == [
        "rdkit_random",
        "qwen_0p6b_bigdata",
    ]
    assert result.loc[
        result["family"] == "qwen_0p6b_bigdata", "display_label"
    ].item() == "Qwen 0.6B bigdata"
    assert frame.loc[1, "display_label"] == "old label"


def test_public_qwen_family_ids_and_labels_match_requested_shortlist() -> None:
    dashboard = load_dashboard()
    expected = {
        "qwen_0p6b_bigdata": "Qwen 0.6B bigdata",
        "qwen_0p6b_bigdata_to_revisited": "Qwen 0.6B bigdata→revisited",
        "qwen_0p6b_fsq": "Qwen 0.6B fsq",
        "qwen_0p6b_fsq_bigdata_pretrain": "Qwen 0.6B fsq+bigdata-pretrain",
        "qwen_1p7b_bigdata": "Qwen 1.7B bigdata",
        "qwen_1p7b_bigdata_to_revisited": "Qwen 1.7B bigdata→revisited",
        "qwen_1p7b_fsq": "Qwen 1.7B fsq",
        "qwen_1p7b_fsq_bigdata_pretrain": "Qwen 1.7B fsq+bigdata-pretrain",
        "qwen_1p7b_revisited": "Qwen 1.7B revisited",
        "qwen_4b_bigdata": "Qwen 4B bigdata",
        "qwen_4b_revisited": "Qwen 4B revisited",
    }
    assert dashboard.QWEN_DASHBOARD_LABELS == expected
