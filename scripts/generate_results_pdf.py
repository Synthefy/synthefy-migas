"""Generate a multi-page PDF of backtest results tables."""

import json
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib

matplotlib.rcParams["font.family"] = "monospace"

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_PATH = RESULTS_DIR / "backtest_results.pdf"

MODELS = {
    "Chronos-2": "chronos2_mv_stride32.json",
    "Toto": "toto_mv_stride32.json",
    "TabPFN": "tabpfn_ts_stride32_v2.0.json",
    "Migas-1.0": "migas_stride32.json",
    "Migas-1.5": "migas_1.5_stride32.json",
}

DATASETS = [
    "NO_1_daily_hydro_reservoir_features",
    "NO_2_daily_hydro_reservoir_features",
    "SE_1_daily_hydro_reservoir_features",
    "SE_3_daily_hydro_reservoir_features",
]
DS_SHORT = {
    "NO_1_daily_hydro_reservoir_features": "NO_1",
    "NO_2_daily_hydro_reservoir_features": "NO_2",
    "SE_1_daily_hydro_reservoir_features": "SE_1",
    "SE_3_daily_hydro_reservoir_features": "SE_3",
}
CTX_LENS = ["32", "64", "128", "256", "384", "512"]
MODES = ["univariate", "no_leak", "planned_leak", "all_leak"]
MODE_LABELS = {
    "univariate": "Univariate",
    "no_leak": "No Leak",
    "planned_leak": "Planned Leak",
    "all_leak": "All Leak",
}

BEST_COLOR = "#C8E6C9"
HEADER_COLOR = "#37474F"
HEADER_TEXT = "white"

MODE_COLORS = {
    "univariate":   ("#E3F2FD", "#BBDEFB"),   # blue tint
    "no_leak":      ("#FFF8E1", "#FFECB3"),   # amber tint
    "planned_leak": ("#FBE9E7", "#FFCCBC"),   # deep-orange tint
    "all_leak":     ("#F3E5F5", "#E1BEE7"),   # purple tint
}
SECTION_COLORS = {
    "univariate":   "#1565C0",
    "no_leak":      "#F57F17",
    "planned_leak": "#BF360C",
    "all_leak":     "#6A1B9A",
}
SECTION_TEXT_COLOR = "white"


def load_data():
    data = {}
    for name, fname in MODELS.items():
        with open(RESULTS_DIR / fname) as f:
            data[name] = json.load(f)
    return data


def get_val(data, model, ds, ctx, mode, metric):
    d = data[model]
    try:
        return d[ds][ctx][mode][metric]
    except KeyError:
        return None


def fmt(v):
    return f"{v:.4f}" if v is not None else "—"


def draw_table(ax, col_labels, row_data, title, col_widths=None,
               bold_cols=None):
    """Draw a styled table on ax.

    row_data: list of tuples:
        (cell_texts, is_section_header, mode_key, local_idx)
        - mode_key: which scenario this row belongs to (for coloring)
        - local_idx: row index within the mode group (for alternating shade)
    bold_cols: for each row, the set of column indices to highlight as best
    """
    ax.set_axis_off()
    n_cols = len(col_labels)

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    cell_texts = [r[0] for r in row_data]
    tbl = ax.table(
        cellText=cell_texts,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        colWidths=col_widths,
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#BDBDBD")
        cell.set_linewidth(0.5)
        cell.set_height(0.045)

        if row == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color=HEADER_TEXT, fontweight="bold", fontsize=8)
            cell.set_height(0.055)
        else:
            actual_row = row - 1
            rd = row_data[actual_row]
            is_section = rd[1]
            mode_key = rd[2]
            local_idx = rd[3]

            if is_section:
                cell.set_facecolor(SECTION_COLORS.get(mode_key, "#455A64"))
                cell.set_text_props(fontweight="bold", fontsize=8,
                                    color=SECTION_TEXT_COLOR)
                cell.set_height(0.04)
            else:
                c1, c2 = MODE_COLORS.get(mode_key, ("#FFFFFF", "#F5F5F5"))
                bg = c1 if local_idx % 2 == 0 else c2
                cell.set_facecolor(bg)

                if bold_cols and actual_row in bold_cols and col in bold_cols[actual_row]:
                    cell.set_facecolor(BEST_COLOR)
                    cell.set_text_props(fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20, color="#212121")


def _add_legend(fig):
    """Add a color-coded scenario legend strip at the bottom of the figure."""
    from matplotlib.patches import FancyBboxPatch
    labels = [
        ("Univariate", MODE_COLORS["univariate"][0], SECTION_COLORS["univariate"]),
        ("No Leak", MODE_COLORS["no_leak"][0], SECTION_COLORS["no_leak"]),
        ("Planned Leak", MODE_COLORS["planned_leak"][0], SECTION_COLORS["planned_leak"]),
        ("All Leak", MODE_COLORS["all_leak"][0], SECTION_COLORS["all_leak"]),
        ("Best", BEST_COLOR, None),
    ]
    n = len(labels)
    spacing = 0.17
    x_start = 0.5 - (n - 1) * spacing / 2
    for i, (label, bg, border) in enumerate(labels):
        x = x_start + i * spacing
        fig.patches.append(FancyBboxPatch(
            (x - 0.025, 0.012), 0.015, 0.015,
            boxstyle="round,pad=0.002",
            facecolor=bg, edgecolor=border or "#888888",
            linewidth=0.8, transform=fig.transFigure, clip_on=False,
        ))
        fig.text(x - 0.005, 0.019, label, fontsize=6.5, va="center",
                 color="#424242")


def page_averaged(pdf, data, metric_key, metric_label):
    """Page 1/2: averaged metric across all datasets, by mode × ctx_len."""
    model_names = list(MODELS.keys())
    col_labels = ["Forecast Mode", "Context Len"] + model_names

    rows = []
    bold_map = {}
    row_idx = 0

    for mode in MODES:
        for local_idx, ctx in enumerate(CTX_LENS):
            avgs = []
            for m in model_names:
                vals = []
                for ds in DATASETS:
                    v = get_val(data, m, ds, ctx, mode, metric_key)
                    if v is not None:
                        vals.append(v)
                avgs.append(np.mean(vals) if vals else None)

            numeric = [(i, v) for i, v in enumerate(avgs) if v is not None]
            if numeric:
                best_i = min(numeric, key=lambda x: x[1])[0]
                bold_map[row_idx] = {best_i + 2}

            row_texts = [MODE_LABELS[mode], ctx] + [fmt(v) for v in avgs]
            rows.append((row_texts, False, mode, local_idx))
            row_idx += 1

    n_cols = len(col_labels)
    col_widths = [0.14, 0.10] + [0.152] * len(model_names)
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]

    fig, ax = plt.subplots(figsize=(13, 8.5))
    draw_table(ax, col_labels, rows,
               f"{metric_label} — Averaged Across All Datasets (Normalized)",
               col_widths=col_widths, bold_cols=bold_map)

    _add_legend(fig)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _compute_wins(data, metric_key):
    """For each (ds, ctx, mode), find the model with the lowest value.

    Returns a dict mapping each model name to a list of
    (ds, ctx, mode) tuples it won.
    """
    model_names = list(MODELS.keys())
    wins = {m: [] for m in model_names}

    for ds in DATASETS:
        for ctx in CTX_LENS:
            for mode in MODES:
                vals = {}
                for m in model_names:
                    v = get_val(data, m, ds, ctx, mode, metric_key)
                    if v is not None:
                        vals[m] = v
                if vals:
                    best_model = min(vals, key=vals.get)
                    wins[best_model].append((ds, ctx, mode))
    return wins


def _wins_table(ax, title, row_labels, model_names, counts, totals,
                row_colors=None):
    """Draw a wins table: rows = categories, cols = models, cells = 'W / T'."""
    col_labels = [""] + model_names
    col_widths = [0.18] + [0.164] * len(model_names)
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]

    cell_texts = []
    bold_map = {}
    for r_idx, label in enumerate(row_labels):
        row = [label]
        best_count = -1
        best_col = None
        for c_idx, m in enumerate(model_names):
            w = counts[r_idx][m]
            t = totals[r_idx]
            row.append(f"{w} / {t}")
            if w > best_count:
                best_count = w
                best_col = c_idx
        cell_texts.append(row)
        if best_col is not None and best_count > 0:
            bold_map[r_idx] = {best_col + 1}

    ax.set_axis_off()
    tbl = ax.table(
        cellText=cell_texts,
        colLabels=col_labels,
        cellLoc="center", colLoc="center",
        loc="upper center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#BDBDBD")
        cell.set_linewidth(0.5)
        cell.set_height(0.055)
        if row == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color=HEADER_TEXT, fontweight="bold", fontsize=8)
            cell.set_height(0.065)
        else:
            actual_row = row - 1
            bg = "#F5F5F5" if actual_row % 2 == 0 else "#FFFFFF"
            if row_colors and actual_row < len(row_colors) and row_colors[actual_row]:
                bg = row_colors[actual_row]
            cell.set_facecolor(bg)
            if actual_row in bold_map and col in bold_map[actual_row]:
                cell.set_facecolor(BEST_COLOR)
                cell.set_text_props(fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=14, color="#212121")


def pages_wins_summary(pdf, data):
    """Generate win-count summary pages (MAE and MSE)."""
    model_names = list(MODELS.keys())
    non_uni_modes = [m for m in MODES if m != "univariate"]

    for metric_key, metric_short in [("MAE (mean)", "MAE"), ("MSE (mean)", "MSE")]:
        wins = _compute_wins(data, metric_key)

        # ── Page 1: All modes (including univariate) ──
        fig = plt.figure(figsize=(13, 16))
        gs = fig.add_gridspec(5, 1, hspace=0.55, top=0.94, bottom=0.03)

        # 1) Overall
        ax0 = fig.add_subplot(gs[0])
        total_possible = len(DATASETS) * len(CTX_LENS) * len(MODES)
        counts_overall = [{m: len(wins[m]) for m in model_names}]
        _wins_table(ax0, f"Overall Wins ({metric_short})",
                    ["All"], model_names, counts_overall, [total_possible])

        # 2) By context length
        ax1 = fig.add_subplot(gs[1])
        row_labels_ctx = [f"ctx {c}" for c in CTX_LENS]
        counts_ctx = []
        totals_ctx = []
        for ctx in CTX_LENS:
            c = {m: sum(1 for _, wc, _ in wins[m] if wc == ctx) for m in model_names}
            counts_ctx.append(c)
            totals_ctx.append(len(DATASETS) * len(MODES))
        _wins_table(ax1, f"Wins by Context Length ({metric_short})",
                    row_labels_ctx, model_names, counts_ctx, totals_ctx)

        # 3) By leak scenario
        ax2 = fig.add_subplot(gs[2])
        row_labels_mode = [MODE_LABELS[m] for m in MODES]
        counts_mode = []
        totals_mode = []
        row_colors_mode = []
        for mode in MODES:
            c = {m: sum(1 for _, _, wm in wins[m] if wm == mode) for m in model_names}
            counts_mode.append(c)
            totals_mode.append(len(DATASETS) * len(CTX_LENS))
            row_colors_mode.append(MODE_COLORS[mode][0])
        _wins_table(ax2, f"Wins by Leak Scenario ({metric_short})",
                    row_labels_mode, model_names, counts_mode, totals_mode,
                    row_colors=row_colors_mode)

        # 4) By dataset
        ax3 = fig.add_subplot(gs[3])
        row_labels_ds = [DS_SHORT[ds] for ds in DATASETS]
        counts_ds = []
        totals_ds = []
        for ds in DATASETS:
            c = {m: sum(1 for wd, _, _ in wins[m] if wd == ds) for m in model_names}
            counts_ds.append(c)
            totals_ds.append(len(CTX_LENS) * len(MODES))
        _wins_table(ax3, f"Wins by Dataset ({metric_short})",
                    row_labels_ds, model_names, counts_ds, totals_ds)

        # 5) By context length × leak scenario
        ax4 = fig.add_subplot(gs[4])
        row_labels_cx_mode = []
        counts_cx_mode = []
        totals_cx_mode = []
        row_colors_cx = []
        for ctx in CTX_LENS:
            for mode in MODES:
                row_labels_cx_mode.append(f"ctx {ctx} / {MODE_LABELS[mode]}")
                c = {m: sum(1 for _, wc, wm in wins[m] if wc == ctx and wm == mode)
                     for m in model_names}
                counts_cx_mode.append(c)
                totals_cx_mode.append(len(DATASETS))
                row_colors_cx.append(MODE_COLORS[mode][0])
        _wins_table(ax4, f"Wins by Context Length × Scenario ({metric_short})",
                    row_labels_cx_mode, model_names, counts_cx_mode,
                    totals_cx_mode, row_colors=row_colors_cx)

        fig.suptitle(f"Model Win Counts — {metric_short} (All Modes)",
                     fontsize=15, fontweight="bold", color="#212121")
        fig.text(0.5, 0.01,
                 "Win = lowest metric value for that cell  |  "
                 "Format: wins / possible  |  Green = most wins in row",
                 ha="center", fontsize=7, color="#757575")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2: Excluding univariate (fairer comparison for Migas-1.5) ──
        fig2 = plt.figure(figsize=(13, 16))
        gs2 = fig2.add_gridspec(5, 1, hspace=0.55, top=0.94, bottom=0.03)

        # Filter wins to non-univariate only
        wins_nu = {m: [(d, c, mo) for d, c, mo in wins[m] if mo != "univariate"]
                   for m in model_names}

        # 1) Overall (excl. univariate)
        ax0 = fig2.add_subplot(gs2[0])
        total_nu = len(DATASETS) * len(CTX_LENS) * len(non_uni_modes)
        counts_nu = [{m: len(wins_nu[m]) for m in model_names}]
        _wins_table(ax0, f"Overall Wins excl. Univariate ({metric_short})",
                    ["All (excl. uni)"], model_names, counts_nu, [total_nu])

        # 2) By context length (excl. univariate)
        ax1 = fig2.add_subplot(gs2[1])
        row_labels_ctx = [f"ctx {c}" for c in CTX_LENS]
        counts_ctx_nu = []
        totals_ctx_nu = []
        for ctx in CTX_LENS:
            c = {m: sum(1 for _, wc, _ in wins_nu[m] if wc == ctx)
                 for m in model_names}
            counts_ctx_nu.append(c)
            totals_ctx_nu.append(len(DATASETS) * len(non_uni_modes))
        _wins_table(ax1, f"Wins by Context Length excl. Univariate ({metric_short})",
                    row_labels_ctx, model_names, counts_ctx_nu, totals_ctx_nu)

        # 3) By leak scenario (excl. univariate)
        ax2 = fig2.add_subplot(gs2[2])
        row_labels_mode_nu = [MODE_LABELS[m] for m in non_uni_modes]
        counts_mode_nu = []
        totals_mode_nu = []
        row_colors_mode_nu = []
        for mode in non_uni_modes:
            c = {m: sum(1 for _, _, wm in wins_nu[m] if wm == mode)
                 for m in model_names}
            counts_mode_nu.append(c)
            totals_mode_nu.append(len(DATASETS) * len(CTX_LENS))
            row_colors_mode_nu.append(MODE_COLORS[mode][0])
        _wins_table(ax2, f"Wins by Leak Scenario excl. Univariate ({metric_short})",
                    row_labels_mode_nu, model_names, counts_mode_nu,
                    totals_mode_nu, row_colors=row_colors_mode_nu)

        # 4) By dataset (excl. univariate)
        ax3 = fig2.add_subplot(gs2[3])
        row_labels_ds = [DS_SHORT[ds] for ds in DATASETS]
        counts_ds_nu = []
        totals_ds_nu = []
        for ds in DATASETS:
            c = {m: sum(1 for wd, _, _ in wins_nu[m] if wd == ds)
                 for m in model_names}
            counts_ds_nu.append(c)
            totals_ds_nu.append(len(CTX_LENS) * len(non_uni_modes))
        _wins_table(ax3, f"Wins by Dataset excl. Univariate ({metric_short})",
                    row_labels_ds, model_names, counts_ds_nu, totals_ds_nu)

        # 5) By context length × leak scenario (excl. univariate)
        ax4 = fig2.add_subplot(gs2[4])
        row_labels_cx_nu = []
        counts_cx_nu = []
        totals_cx_nu = []
        row_colors_cx_nu = []
        for ctx in CTX_LENS:
            for mode in non_uni_modes:
                row_labels_cx_nu.append(f"ctx {ctx} / {MODE_LABELS[mode]}")
                c = {m: sum(1 for _, wc, wm in wins_nu[m]
                            if wc == ctx and wm == mode)
                     for m in model_names}
                counts_cx_nu.append(c)
                totals_cx_nu.append(len(DATASETS))
                row_colors_cx_nu.append(MODE_COLORS[mode][0])
        _wins_table(ax4,
                    f"Wins by Ctx × Scenario excl. Univariate ({metric_short})",
                    row_labels_cx_nu, model_names, counts_cx_nu,
                    totals_cx_nu, row_colors=row_colors_cx_nu)

        fig2.suptitle(
            f"Model Win Counts — {metric_short} (Excluding Univariate)",
            fontsize=15, fontweight="bold", color="#212121")
        fig2.text(0.5, 0.01,
                  "Univariate mode excluded for fair comparison "
                  "(Migas-1.5 has no univariate)  |  "
                  "Green = most wins in row",
                  ha="center", fontsize=7, color="#757575")
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)


def page_per_ctx(pdf, data, ctx):
    """One page per context length: grouped by mode, then by dataset."""
    model_names = list(MODELS.keys())
    col_labels = ["Dataset"] + model_names
    metrics = [("MAE (mean)", "MAE"), ("MSE (mean)", "MSE")]

    for metric_key, metric_short in metrics:
        rows = []
        bold_map = {}
        row_idx = 0

        for mode in MODES:
            rows.append(([MODE_LABELS[mode]] + [""] * len(model_names),
                         True, mode, 0))
            row_idx += 1

            for local_idx, ds in enumerate(DATASETS):
                vals = [get_val(data, m, ds, ctx, mode, metric_key) for m in model_names]
                numeric = [(i, v) for i, v in enumerate(vals) if v is not None]
                if numeric:
                    best_i = min(numeric, key=lambda x: x[1])[0]
                    bold_map[row_idx] = {best_i + 1}

                row_texts = [DS_SHORT[ds]] + [fmt(v) for v in vals]
                rows.append((row_texts, False, mode, local_idx))
                row_idx += 1

            avg_vals = []
            for m in model_names:
                m_vals = [get_val(data, m, ds, ctx, mode, metric_key) for ds in DATASETS]
                m_vals = [v for v in m_vals if v is not None]
                avg_vals.append(np.mean(m_vals) if m_vals else None)
            numeric = [(i, v) for i, v in enumerate(avg_vals) if v is not None]
            if numeric:
                best_i = min(numeric, key=lambda x: x[1])[0]
                bold_map[row_idx] = {best_i + 1}
            row_texts = ["AVG"] + [fmt(v) for v in avg_vals]
            rows.append((row_texts, False, mode, len(DATASETS)))
            row_idx += 1

        col_widths = [0.12] + [0.176] * len(model_names)
        total = sum(col_widths)
        col_widths = [w / total for w in col_widths]

        fig, ax = plt.subplots(figsize=(13, 8.5))
        draw_table(ax, col_labels, rows,
                   f"{metric_short} by Dataset — Context Length {ctx}",
                   col_widths=col_widths, bold_cols=bold_map)
        _add_legend(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


MODEL_BG = {
    "Chronos-2": ("#E8F5E9", "#C8E6C9"),
    "Toto":      ("#E3F2FD", "#BBDEFB"),
    "TabPFN":    ("#FFF3E0", "#FFE0B2"),
    "Migas-1.0": ("#F3E5F5", "#E1BEE7"),
    "Migas-1.5": ("#E0F7FA", "#B2EBF2"),
}


def page_per_model(pdf, data):
    """One page per model: rows = ctx lengths, columns = scenarios.

    Two tables per page (MAE on top, MSE on bottom).
    Values are averaged across all 4 datasets.
    """
    mode_labels_short = [MODE_LABELS[m] for m in MODES]
    col_labels = ["Context Len"] + mode_labels_short

    col_widths = [0.16] + [0.21] * len(MODES)
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]

    for model_name in MODELS:
        fig, (ax_mae, ax_mse) = plt.subplots(2, 1, figsize=(11, 8.5))

        for ax, (metric_key, metric_short) in zip(
            [ax_mae, ax_mse],
            [("MAE (mean)", "MAE"), ("MSE (mean)", "MSE")],
        ):
            rows = []
            bold_map = {}

            for local_idx, ctx in enumerate(CTX_LENS):
                vals = []
                for mode in MODES:
                    per_ds = [get_val(data, model_name, ds, ctx, mode, metric_key)
                              for ds in DATASETS]
                    per_ds = [v for v in per_ds if v is not None]
                    vals.append(np.mean(per_ds) if per_ds else None)

                numeric = [(i, v) for i, v in enumerate(vals) if v is not None]
                if numeric:
                    best_i = min(numeric, key=lambda x: x[1])[0]
                    bold_map[local_idx] = {best_i + 1}

                row_texts = [ctx] + [fmt(v) for v in vals]
                rows.append((row_texts, False, None, local_idx))

            ax.set_axis_off()
            cell_texts = [r[0] for r in rows]
            tbl = ax.table(
                cellText=cell_texts,
                colLabels=col_labels,
                cellLoc="center",
                colLoc="center",
                loc="upper center",
                colWidths=col_widths,
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)

            c1, c2 = MODEL_BG.get(model_name, ("#FFFFFF", "#F5F5F5"))

            for (row, col), cell in tbl.get_celld().items():
                cell.set_edgecolor("#BDBDBD")
                cell.set_linewidth(0.5)
                cell.set_height(0.09)

                if row == 0:
                    if col == 0:
                        cell.set_facecolor(HEADER_COLOR)
                        cell.set_text_props(color=HEADER_TEXT, fontweight="bold",
                                            fontsize=9)
                    else:
                        mode_key = MODES[col - 1]
                        cell.set_facecolor(SECTION_COLORS[mode_key])
                        cell.set_text_props(color="white", fontweight="bold",
                                            fontsize=9)
                    cell.set_height(0.1)
                else:
                    actual_row = row - 1
                    bg = c1 if actual_row % 2 == 0 else c2
                    cell.set_facecolor(bg)

                    if actual_row in bold_map and col in bold_map[actual_row]:
                        cell.set_facecolor(BEST_COLOR)
                        cell.set_text_props(fontweight="bold")

            ax.set_title(f"{metric_short} (avg across datasets)",
                         fontsize=11, fontweight="bold", pad=12, color="#424242")

        fig.suptitle(model_name, fontsize=15, fontweight="bold", y=0.98,
                     color="#212121")
        fig.subplots_adjust(hspace=0.35, top=0.92)
        fig.text(0.5, 0.02, "Green = best scenario for that context length",
                 ha="center", fontsize=7, color="#757575")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


DS_ROW_COLORS = {
    "NO_1": ("#E8EAF6", "#C5CAE9"),
    "NO_2": ("#E0F2F1", "#B2DFDB"),
    "SE_1": ("#FFF8E1", "#FFECB3"),
    "SE_3": ("#FCE4EC", "#F8BBD0"),
}
DS_SECTION_HDR = {
    "NO_1": "#283593",
    "NO_2": "#00695C",
    "SE_1": "#F57F17",
    "SE_3": "#AD1457",
}


def page_per_model_detailed(pdf, data):
    """One page per (model × metric): rows grouped by dataset, cols = scenarios.

    Each page has dataset section headers followed by ctx-length rows.
    """
    mode_labels_short = [MODE_LABELS[m] for m in MODES]
    col_labels = ["Dataset / Ctx"] + mode_labels_short

    col_widths = [0.18] + [0.205] * len(MODES)
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]

    for model_name in MODELS:
        for metric_key, metric_short in [("MAE (mean)", "MAE"),
                                         ("MSE (mean)", "MSE")]:
            rows = []
            bold_map = {}
            row_idx = 0

            for ds in DATASETS:
                short = DS_SHORT[ds]
                rows.append(([short] + [""] * len(MODES),
                             True, short, 0))
                row_idx += 1

                for local_idx, ctx in enumerate(CTX_LENS):
                    vals = []
                    for mode in MODES:
                        v = get_val(data, model_name, ds, ctx, mode, metric_key)
                        vals.append(v)

                    numeric = [(i, v) for i, v in enumerate(vals)
                               if v is not None]
                    if numeric:
                        best_i = min(numeric, key=lambda x: x[1])[0]
                        bold_map[row_idx] = {best_i + 1}

                    row_texts = [f"  ctx {ctx}"] + [fmt(v) for v in vals]
                    rows.append((row_texts, False, short, local_idx))
                    row_idx += 1

            fig, ax = plt.subplots(figsize=(11, 10))
            ax.set_axis_off()

            cell_texts = [r[0] for r in rows]
            tbl = ax.table(
                cellText=cell_texts,
                colLabels=col_labels,
                cellLoc="center",
                colLoc="center",
                loc="upper center",
                colWidths=col_widths,
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)

            for (row, col), cell in tbl.get_celld().items():
                cell.set_edgecolor("#BDBDBD")
                cell.set_linewidth(0.5)
                cell.set_height(0.032)

                if row == 0:
                    if col == 0:
                        cell.set_facecolor(HEADER_COLOR)
                        cell.set_text_props(color=HEADER_TEXT, fontweight="bold",
                                            fontsize=8)
                    else:
                        mode_key = MODES[col - 1]
                        cell.set_facecolor(SECTION_COLORS[mode_key])
                        cell.set_text_props(color="white", fontweight="bold",
                                            fontsize=8)
                    cell.set_height(0.038)
                else:
                    actual_row = row - 1
                    rd = rows[actual_row]
                    is_section = rd[1]
                    ds_key = rd[2]
                    local_idx = rd[3]

                    if is_section:
                        cell.set_facecolor(
                            DS_SECTION_HDR.get(ds_key, "#455A64"))
                        cell.set_text_props(fontweight="bold", fontsize=8,
                                            color="white")
                        cell.set_height(0.030)
                    else:
                        c1, c2 = DS_ROW_COLORS.get(ds_key,
                                                    ("#FFFFFF", "#F5F5F5"))
                        cell.set_facecolor(c1 if local_idx % 2 == 0 else c2)

                        if (actual_row in bold_map
                                and col in bold_map[actual_row]):
                            cell.set_facecolor(BEST_COLOR)
                            cell.set_text_props(fontweight="bold")

            ax.set_title(f"{model_name} — {metric_short} by Dataset × Scenario",
                         fontsize=13, fontweight="bold", pad=16, color="#212121")

            fig.text(0.5, 0.02,
                     "Green = best scenario for that row  |  "
                     "Column headers colored by scenario",
                     ha="center", fontsize=7, color="#757575")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main():
    data = load_data()

    # 1) Overview: averaged MAE & MSE across all datasets (2 pages)
    p1 = RESULTS_DIR / "1_overview_averaged.pdf"
    with PdfPages(str(p1)) as pdf:
        page_averaged(pdf, data, "MAE (mean)", "MAE (Mean)")
        page_averaged(pdf, data, "MSE (mean)", "MSE (Mean)")
    print(f"  {p1.name}  ({2} pages)")

    # 2) Per context length: wins summary + scenario→dataset breakdown
    p2 = RESULTS_DIR / "2_per_context_length.pdf"
    with PdfPages(str(p2)) as pdf:
        pages_wins_summary(pdf, data)
        for ctx in CTX_LENS:
            page_per_ctx(pdf, data, ctx)
    n_pages_2 = 4 + len(CTX_LENS) * 2
    print(f"  {p2.name}  ({n_pages_2} pages)")

    # 3) Per model (averaged): ctx rows × scenario cols (4 pages)
    p3 = RESULTS_DIR / "3_per_model_averaged.pdf"
    with PdfPages(str(p3)) as pdf:
        page_per_model(pdf, data)
    print(f"  {p3.name}  ({len(MODELS)} pages)")

    # 4) Per model (detailed): dataset-grouped rows × scenario cols (8 pages)
    p4 = RESULTS_DIR / "4_per_model_by_dataset.pdf"
    with PdfPages(str(p4)) as pdf:
        page_per_model_detailed(pdf, data)
    print(f"  {p4.name}  ({len(MODELS) * 2} pages)")

    print(f"\nAll PDFs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
