"""A self-contained HTML QC report: what was removed, and the plots to justify it.

One file, no assets beside it, openable anywhere. It pulls together what the
pipeline recorded as it ran -- the filter log, the gates and the channels they
were drawn on -- so the report is a record of what actually happened rather
than something assembled by hand afterwards.
"""

from __future__ import annotations

import base64
import html
import io
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ._util import subsample_indices

__all__ = ["gating_pdf", "report"]

#: Sections drawn when none are named, in order. Each is skipped when the data
#: carries nothing for it.
SECTIONS = ("summary", "filters", "gates")

_CSS = """
:root { color-scheme: light dark; }
body { font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 0; padding: 2.5rem 1.5rem; background: #fbfbfc; color: #1a1a1a; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.05rem; margin: 2.5rem 0 .75rem; padding-bottom: .3rem;
     border-bottom: 1px solid #e3e3e7; }
p.sub { color: #6a6a72; margin: 0 0 2rem; font-size: .85rem; }
p.note { color: #6a6a72; font-size: .82rem; margin: .3rem 0 1rem; }
table { border-collapse: collapse; font-size: .82rem; margin: .5rem 0 1rem; width: 100%; }
th, td { text-align: right; padding: .35rem .6rem; border-bottom: 1px solid #ececf0; }
th:first-child, td:first-child { text-align: left; }
thead th { background: #f3f3f6; font-weight: 600; }
tbody tr:hover { background: #f7f7fa; }
figure { margin: 0 0 1.5rem; }
figure img { max-width: 100%; height: auto; display: block; border: 1px solid #ececf0;
             border-radius: 4px; background: #fff; }
figcaption { color: #6a6a72; font-size: .8rem; margin-top: .35rem; }
.scroll { overflow-x: auto; }
.empty { color: #8a8a92; font-style: italic; }
@media (prefers-color-scheme: dark) {
  body { background: #16161a; color: #e8e8ea; }
  h2 { border-color: #2c2c33; } thead th { background: #22222a; }
  th, td { border-color: #26262d; } tbody tr:hover { background: #1e1e25; }
  figure img { border-color: #2c2c33; background: #fff; }
}
"""


def _encode(fig, dpi: int) -> str:
    """A matplotlib figure as an inline data URI, so the report is one file."""
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _figure(fig, caption: str, dpi: int) -> str:
    return (
        f'<figure><img src="{_encode(fig, dpi)}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _table(frame: pd.DataFrame, *, index: bool = True) -> str:
    if frame.empty:
        return '<p class="empty">nothing recorded</p>'
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda v: "" if pd.isna(v) else f"{v:,.2f}")
        elif pd.api.types.is_integer_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda v: f"{v:,}")
    return '<div class="scroll">' + formatted.to_html(index=index, border=0, escape=True) + "</div>"


def _heading(name: str) -> str:
    return f"<h2>{html.escape(name)}</h2>"


def report(
    adata: ad.AnnData,
    path: str | os.PathLike,
    *,
    title: str = "cytopy QC report",
    sections: Sequence[str] | None = None,
    dpi: int = 110,
    subsample: int | None = 500_000,
    max_gates: int = 12,
    seed: int = 0,
) -> Path:
    """Write an HTML report of what a pipeline did to this data.

    Parameters
    ----------
    adata
        The AnnData at the end of the pipeline. Everything drawn comes from
        what the earlier steps recorded on it, so the report reflects the run
        rather than being reconstructed.
    path
        HTML file to write. The plots are embedded, so it stands alone.
    title
        Heading for the report.
    sections
        Which of :data:`SECTIONS` to include, in order. ``None`` includes
        every one the data has something for.
    dpi
        Resolution of the embedded figures.
    subsample
        Events to draw the biaxial plots from, sampled once and reused.
        ``None`` draws all of them. The default keeps a report on a run of
        millions of events quick without visibly changing a density.
    max_gates
        Most gate panels to draw, newest first.
    seed
        Seed for that subsample, so a report redrawn from the same data shows
        the same events.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    ValueError
        If ``sections`` names something that is not in :data:`SECTIONS`.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)

    chosen = list(SECTIONS if sections is None else sections)
    unknown = [s for s in chosen if s not in SECTIONS]
    if unknown:
        raise ValueError(f"unknown report section(s) {unknown}; choose from {list(SECTIONS)}")

    plotted = _thin_events(adata, subsample, seed)
    builders = {
        "summary": lambda: _summary_section(adata),
        "filters": lambda: _filters_section(adata, dpi),
        "gates": lambda: _gates_section(adata, plotted, dpi, max_gates),
    }
    body = "".join(filter(None, (builders[name]() for name in chosen)))

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    document = (
        f"<!doctype html><html lang=en><head><meta charset=utf-8>"
        f'<meta name=viewport content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><main>"
        f"<h1>{html.escape(title)}</h1>"
        f'<p class="sub">{adata.n_obs:,} events &times; {adata.n_vars} channels &middot; '
        f"generated {html.escape(stamp)}</p>{body}</main></body></html>"
    )
    path = Path(path)
    path.write_text(document, encoding="utf-8")
    return path


def _thin_events(
    adata: ad.AnnData | None, subsample: int | None, seed: int = 0
) -> ad.AnnData | None:
    """A stable subsample for the biaxial plots; a density does not need every event.

    Returns a view rather than a copy: the report only reads from it.
    """
    if adata is None:
        return adata
    take = subsample_indices(adata.n_obs, subsample, rng=np.random.default_rng(seed))
    return adata if take is None else adata[take]


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def _summary_section(adata: ad.AnnData) -> str:
    rows = [
        {"": "events", "value": f"{adata.n_obs:,}"},
        {"": "channels", "value": str(adata.n_vars)},
    ]
    if "sample" in adata.obs:
        counts = adata.obs["sample"].astype(str).value_counts()
        for name, n in counts.items():
            rows.append({"": f"sample {name}", "value": f"{int(n):,}"})
    info = adata.uns.get("cytopy", {})
    for label, key in [
        ("compensated layer", "compensated_layer"),
        ("spillover from", "spillover_source"),
        ("arcsinh layer", "asinh_layer"),
        ("logicle layer", "logicle_layer"),
    ]:
        if info.get(key):
            rows.append({"": label, "value": str(info[key])})
    frame = pd.DataFrame(rows).set_index("")
    return _heading("Summary") + _table(frame)


def _filters_section(adata: ad.AnnData, dpi: int) -> str:
    from .filters import filter_log

    log = filter_log(adata)
    if log.empty:
        return ""
    out = _heading("What was removed")
    out += (
        '<p class="note">Every step that dropped events, in the order they ran. '
        "The counts are recorded as each step runs, so they hold even though the "
        "intermediate objects are gone.</p>"
    )
    out += _table(log.set_index("step"))
    out += _figure(_waterfall(log), "events surviving each step", dpi)
    return out


def _waterfall(log: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(log) + 1.4))
    labels = ["start", *log["step"].tolist()]
    counts = [int(log["n_before"].iloc[0]), *log["n_after"].tolist()]
    positions = np.arange(len(labels))
    ax.barh(positions, counts, color="#4c78a8", height=0.6)
    for y, n in zip(positions, counts):
        ax.annotate(
            f"{n:,}", xy=(n, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8
        )
    ax.set_yticks(positions, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("events remaining", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(counts) * 1.15)
    fig.tight_layout()
    return fig


def _gates_section(adata: ad.AnnData, plotted: ad.AnnData, dpi: int, max_gates: int) -> str:
    from .gating import gate_record, gate_stats
    from .plotting import plot_gate

    gates = adata.uns.get("cytopy", {}).get("gates", {})
    drawable = [n for n in gates if n in adata.obs]
    if not drawable:
        return ""
    rows = []
    records = {name: gate_record(adata, name) for name in drawable}
    for name in drawable:
        record = records[name]
        stats = gate_stats(adata, name, parent=record.parent or None)
        rows.append(
            {
                "gate": name,
                "parent": record.parent,
                "channels": f"{record.x} / {record.y}",
                "n": stats["n"],
                "pct_total": stats["pct_total"],
                "pct_parent": stats.get("pct_parent", float("nan")),
            }
        )
    out = _heading("Gates") + _table(pd.DataFrame(rows).set_index("gate"))

    panels = [n for n in drawable if records[n].has_outline()][-max_gates:]
    for name in panels:
        try:
            ax = plot_gate(plotted, name)
        except (KeyError, ValueError):
            continue
        out += _figure(ax.figure, f"gate {name}, in the plane it was drawn in", dpi)
    if not panels and drawable:
        out += (
            '<p class="note">No outlines were recorded for these gates, so only '
            "the counts are shown. Gates drawn in the viewer record their vertices.</p>"
        )
    return out


def gating_pdf(
    adata: ad.AnnData,
    path: str | os.PathLike,
    *,
    title: str | None = None,
    ncols: int = 3,
    per_page: int = 6,
    subsample: int | None = 200_000,
    dpi: int = 150,
    seed: int = 0,
    **kwargs,
) -> Path:
    """Write the gating hierarchy to a PDF, one plot per gate.

    Each gate gets the biaxial it was actually drawn on, showing the events its
    parent let through, with its own outline over them and the events it kept
    picked out. Gates come out parents first, so the pages read the way the
    gating was done.

    Parameters
    ----------
    adata
        AnnData carrying gates in ``uns['cytopy']['gates']``.
    path
        PDF file to write.
    title
        Heading for the first page. Defaults to the sample name, or the file's.
    ncols
        Plots across a page.
    per_page
        Plots on a page. Rounded up to fill whole rows.
    subsample
        Events to draw each plot from. ``None`` draws all of them; the default
        keeps a hierarchy on a run of millions quick without visibly changing
        a density.
    dpi
        Resolution of the figures.
    seed
        Seed for the subsample, so the same events are drawn each time.
    **kwargs
        Passed to :func:`~cytopy.plot_gate`.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    ValueError
        If nothing has been gated, or no gate recorded the plane it was drawn
        in -- there would be nothing to draw.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from .gating import gate_order, gate_record
    from .plotting import plot_gate

    gates = adata.uns.get("cytopy", {}).get("gates", {})
    order = gate_order(adata)
    drawable = [g for g in order if g in adata.obs and gate_record(adata, g).has_outline()]
    if not drawable:
        raise ValueError(
            "no gates with an outline to draw; gates applied from a mask rather "
            "than drawn record no plane"
        )

    plotted = _thin_events(adata, subsample, seed)
    if title is None:
        title = _default_title(adata)
    rows = int(np.ceil(per_page / ncols))
    page_size = rows * ncols

    path = Path(path)
    with PdfPages(path) as pdf:
        pdf.savefig(_hierarchy_page(adata, gates, order, title), dpi=dpi)
        plt.close("all")
        for start in range(0, len(drawable), page_size):
            chunk = drawable[start : start + page_size]
            fig, axes = plt.subplots(rows, ncols, figsize=(3.6 * ncols, 3.5 * rows), squeeze=False)
            flat = axes.ravel()
            for ax, name in zip(flat, chunk):
                _draw_gate_page(plotted, adata, name, ax, plot_gate, kwargs)
            for ax in flat[len(chunk) :]:
                ax.set_visible(False)
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
    return path


def _depth(gates: dict, name: str) -> int:
    depth = 0
    seen = set()
    while name and name in gates and name not in seen:
        seen.add(name)
        name = str(gates[name].get("parent") or "")
        depth += 1 if name else 0
    return depth


def _default_title(adata: ad.AnnData) -> str:
    if "sample" in adata.obs and adata.n_obs:
        names = adata.obs["sample"].astype(str).unique()
        if len(names) == 1:
            return str(names[0])
    return "gating"


def _draw_gate_page(plotted, adata, name, ax, plot_gate, kwargs) -> None:
    """One gate: its parent's events, its outline, and what it kept.

    The figure itself is :func:`~cytopy.plot_gate`; what this adds is a title
    naming the lineage and the share kept, counted on the *whole* object rather
    than the thinned copy the page is drawn from.
    """
    from .gating import gate_record

    record = gate_record(adata, name)
    parent = record.parent

    n = int(adata.obs[name].sum())
    total = int(adata.obs[parent].sum()) if parent and parent in adata.obs else adata.n_obs
    share = 100 * n / max(total, 1)
    lineage = f"{parent} > " if parent else ""

    try:
        plot_gate(
            plotted,
            name,
            color_by=name if name in plotted.obs else "density",
            title=f"{lineage}{name}\n{n:,} of {total:,}  ({share:.1f}%)",
            ax=ax,
            **kwargs,
        )
    except (KeyError, ValueError) as exc:  # pragma: no cover - defensive
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"{name}\n{exc}", ha="center", va="center", fontsize=7)


def _hierarchy_page(adata, gates, order, title):
    """A contents page: the tree, with counts and frequencies."""
    import matplotlib.pyplot as plt

    from .gating import gate_record

    rows = []
    for name in order:
        parent = gate_record(adata, name).parent
        n = int(adata.obs[name].sum()) if name in adata.obs else 0
        total = int(adata.obs[parent].sum()) if parent and parent in adata.obs else adata.n_obs
        rows.append(
            (
                "    " * _depth(gates, name) + name,
                f"{n:,}",
                f"{100 * n / max(total, 1):.2f}%",
                f"{100 * n / max(adata.n_obs, 1):.2f}%",
            )
        )

    height = 1.6 + 0.24 * len(rows)
    fig, ax = plt.subplots(figsize=(8.3, min(height, 11.0)))
    ax.set_axis_off()
    ax.text(0, 1.0, title, fontsize=15, va="top", weight="bold", transform=ax.transAxes)
    ax.text(
        0,
        0.955,
        f"{adata.n_obs:,} events   {len(rows)} gates",
        fontsize=9,
        va="top",
        color="#666",
        transform=ax.transAxes,
    )
    table = ax.table(
        cellText=rows,
        colLabels=["gate", "events", "% of parent", "% of total"],
        colWidths=[0.5, 0.17, 0.17, 0.16],
        cellLoc="right",
        loc="upper left",
        bbox=[0, 0, 1, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        if col == 0:
            cell.set_text_props(ha="left")
        if row == 0:
            cell.set_text_props(weight="bold")
    return fig
