"""Command-line entry point for reusable conformer report plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from casf_benchmark.visualization.report_plots import (
    ReportPlotConfig,
    generate_report_plots,
    load_plot_data,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate report-style energy and RMSD plots from long-form ligand data."
        )
    )
    parser.add_argument("input", type=Path, help="CSV, Parquet, or SQLite input file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conformer_report_plots"),
        help="Directory for generated PNG files.",
    )
    parser.add_argument(
        "--table",
        default="per_ligand_long",
        help="SQLite table name (default: per_ligand_long).",
    )
    parser.add_argument(
        "--tier",
        help="Optional tier to select, such as chembl_count or fixed.",
    )
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="Method to include; repeat for multiple methods. Defaults to all.",
    )
    parser.add_argument(
        "--reference-method",
        default="CASF crystal",
        help="Method used as the paired energy reference.",
    )
    parser.add_argument(
        "--rmsd-order-method",
        default="ChEMBL3D ground truth",
        help="Method whose best RMSD determines the shared ligand order.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_plot_data(args.input, table=args.table)
    config = ReportPlotConfig(
        reference_method=args.reference_method,
        rmsd_order_method=args.rmsd_order_method,
        tier=args.tier,
        methods=tuple(args.methods) if args.methods else None,
    )
    outputs = generate_report_plots(frame, args.output_dir, config=config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
