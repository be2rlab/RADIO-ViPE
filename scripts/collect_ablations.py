

from __future__ import annotations

import argparse
import csv
from pathlib import Path

CORE_ROWS = [
    ("E9", r"no robust kernel ($\ell_2$)"),
    ("K1", r"fixed Huber ($\alpha{=}1$)"),
    ("K3", r"fixed Geman--McClure ($\alpha{=}{-}2$)"),
    ("K4", r"per-edge semantic ($\alpha$ from $cs_{ij}$)"),
    ("K8", r"multi-view, mean only ($\bar{cs}_i$)"),
    ("E5", r"\textbf{multi-view, mean \& variance (full)}"),
]

CONSUMER_ROWS = [
    ("E0", r"none of (a)--(d)"),
    ("E6", r"full $-$ (a) semantic flow init"),
    ("E7", r"full $-$ (b) graph topology"),
    ("E8", r"full $-$ (c) $E_{\text{embed}}$"),
    ("E9", r"full $-$ (d) adaptive kernel"),
    ("E5", r"\textbf{full}"),
]

CONTROL_ROWS = [
    ("E0", r"none of (a)--(d)"),
    ("E5", r"\textbf{full}"),
]

SENSITIVITY_ROWS = [
    ("S_ts_065", r"$\theta_s = 0.65$"),
    ("S_ts_085", r"$\theta_s = 0.85$"),
    ("S_tm_025", r"$\theta_m = 0.25$"),
    ("S_tm_045", r"$\theta_m = 0.45$"),
    ("E5",       r"\textbf{$\theta_s{=}0.75$, $\theta_m{=}0.35$ (default)}"),
]

TUM_WALKING = [
    "rgbd_dataset_freiburg3_walking_xyz",
    "rgbd_dataset_freiburg3_walking_rpy",
    "rgbd_dataset_freiburg3_walking_halfsphere",
    "rgbd_dataset_freiburg3_walking_static",
]
WALKING_HEADERS = ["fr3/w/xyz", "fr3/w/rpy", "fr3/w/hs", "fr3/w/static"]

TUM_SITTING = [
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_sitting_rpy",
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_sitting_static",
]
SITTING_HEADERS = ["fr3/s/xyz", "fr3/s/rpy", "fr3/s/hs", "fr3/s/static"]

# Subset used by the optional sensitivity tier (see run_ablations.sh TUM_SENS).
TUM_SENS = [TUM_WALKING[0], TUM_WALKING[2]]
SENS_HEADERS = [WALKING_HEADERS[0], WALKING_HEADERS[2]]


def fmt(value: float | None) -> str:
    """
    Scale to the paper's units and render with two decimals.

    ATE is stored in metres and reported in centimetres.
    """
    if value is None:
        return "--"
    return f"{value * 100:.2f}"


def fmt_delta(value: float | None, ref: float | None) -> str:
    if value is None or ref is None:
        return "--"
    delta = (value - ref) * 100
    if abs(delta) < 5e-3:          # rounds to 0.00 at the reported precision
        return "0.00"
    return f"{delta:+.2f}"


def read_ate(results_dir: Path, preset: str) -> dict[str, float]:
    path = Path(results_dir) / preset / "metrics.csv"
    if not path.is_file():
        return {}
    out: dict[str, float] = {}
    with path.open(newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            try:
                out[row[0]] = float(row[1])
            except ValueError:
                continue          # header row or malformed line
    return out


def build_table(
    data: dict[str, dict[str, float]],
    rows: list[tuple[str, str]],
    scenes: list[str],
    caption: str,
    label: str,
    headers: list[str] | None = None,
    higher_is_better: bool = False,
    delta_ref: str | None = None,
) -> str:
    headers = headers or scenes

    averages: dict[str, float | None] = {}
    for preset, _ in rows:
        present = data.get(preset, {})
        vals = [present[s] for s in scenes if s in present]
        averages[preset] = sum(vals) / len(vals) if len(vals) == len(scenes) else None

    finite = [v for v in averages.values() if v is not None]
    best = (max(finite) if higher_is_better else min(finite)) if finite else None
    ref = averages.get(delta_ref) if delta_ref else None

    col_spec = "l" + "c" * len(scenes) + "|c" + ("c" if delta_ref else "")
    head_cells = [rf"\rotatebox{{90}}{{{h}}}" for h in headers]
    head_cells.append(r"\rotatebox{90}{\textbf{Avg}}")
    if delta_ref:
        head_cells.append(r"\rotatebox{90}{$\Delta$}")

    out = [
        r"\begin{table}[t]",
        r"    \small",
        r"    \setlength{\tabcolsep}{1.5pt}",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \centering",
        r"    \fittable{%",   # \fittable is defined in paper/macros_shared.tex
        f"        \\begin{{tabular}}{{{col_spec}}}",
        r"            \toprule",
        "            \\textbf{Variant} & " + " & ".join(head_cells) + r" \\",
        r"            \midrule",
    ]
    for preset, label_text in rows:
        cells = [fmt(data.get(preset, {}).get(s)) for s in scenes]
        avg = averages[preset]
        avg_cell = fmt(avg)
        if best is not None and avg is not None and abs(avg - best) < 1e-12:
            avg_cell = rf"\best{{{avg_cell}}}"
        cells.append(avg_cell)
        if delta_ref:
            cells.append("--" if preset == delta_ref else fmt_delta(avg, ref))
        out.append(f"            {label_text} & " + " & ".join(cells) + r" \\")
    out += [
        r"            \bottomrule",
        r"        \end{tabular}",
        r"    }%",
        r"\end{table}",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, type=Path,
                    help="SLAM run output root (contains <preset>/metrics.csv)")
    ap.add_argument("--tables", required=True, type=Path,
                    help="paper/tables directory to write into")
    args = ap.parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)

    specs = [
        (
            "ablation-core.tex", CORE_ROWS, TUM_WALKING, WALKING_HEADERS,
            "\\textbf{The temporal stability field is what makes the adaptive "
            "kernel work.} ATE (cm $\\downarrow$) on the four dynamic TUM-RGBD "
            "walking sequences. Every row keeps consumers (a)--(c) enabled, so "
            "the robust kernel is the only variable. $\\Delta$ is the change in "
            "average ATE relative to the full system.",
            "tab:ablation-core", "E5",
        ),
        (
            "ablation-consumers.tex", CONSUMER_ROWS, TUM_WALKING, WALKING_HEADERS,
            "\\textbf{Each consumer of the embedding field is necessary.} "
            "Leave-one-out ATE (cm $\\downarrow$) on the four dynamic TUM-RGBD "
            "walking sequences. (a) semantic flow initialization, (b) embedding "
            "graph topology, (c) embedding BA residual, (d) adaptive robust "
            "kernel. $\\Delta$ is relative to the full system.",
            "tab:ablation-consumers", "E5",
        ),
        (
            "ablation-control.tex", CONTROL_ROWS, TUM_SITTING, SITTING_HEADERS,
            "\\textbf{Low-dynamic control.} ATE (cm $\\downarrow$) on the four "
            "TUM-RGBD \\emph{sitting} sequences, where the scene is nearly "
            "static. Confirms the embedding machinery does not cost accuracy "
            "when there is little motion to reject.",
            "tab:ablation-control", "E5",
        ),
        (
            "ablation-sensitivity.tex", SENSITIVITY_ROWS, TUM_SENS, SENS_HEADERS,
            "\\textbf{Threshold sensitivity.} ATE (cm $\\downarrow$) on two "
            "walking sequences, sweeping each stability threshold "
            "independently around its default.",
            "tab:ablation-sensitivity", "E5",
        ),
    ]

    for filename, rows, scenes, headers, caption, label, ref in specs:
        data = {preset: read_ate(args.results, preset) for preset, _ in rows}
        tex = build_table(data, rows, scenes, caption, label, headers,
                          delta_ref=ref)
        (args.tables / filename).write_text(tex)
        print(f"wrote {args.tables / filename}")


if __name__ == "__main__":
    main()
