"""cytopy: cytometry analysis on AnnData, gated interactively in napari."""

from __future__ import annotations

from .beads import (
    BEAD_PANELS,
    bead_baseline,
    bead_channels,
    bead_distance,
    bead_gates,
    dna_channel,
    gate_beads,
    mass_channels,
    normalise_beads,
    plot_bead_gates,
    plot_beads_over_time,
    remove_beads,
)
from .debarcode import (
    BARCODE_SCHEMES,
    apply_cutoffs,
    assign_prelim,
    barcode_key,
    barcode_stats,
    debarcode,
    estimate_cutoffs,
    plot_barcode_events,
    plot_yields,
)
from .density import Axes2D, density_image
from .filters import filter_events, filter_log, record_filter
from .gating import add_gate, gate_stats, polygon_mask
from .io import concat_samples, read_fcs, read_fcs_dir
from .plotting import plot_biaxial, plot_gate
from .report import report
from .scales import AsinhScale, LinearScale, LogicleScale, LogScale, get_scale
from .spillover import (
    read_controls,
    read_spillover,
    spillover_from_controls,
    write_spillover,
)
from .transforms import (
    asinh_transform,
    channel_index,
    compensate,
    estimate_cofactors,
    fluor_channels,
    logicle_transform,
    subsample,
)

__version__ = "0.1.0"

__all__ = [
    "BARCODE_SCHEMES",
    "BEAD_PANELS",
    "AsinhScale",
    "Axes2D",
    "CytoViewer",
    "LinearScale",
    "LogScale",
    "LogicleScale",
    "Panel",
    "add_gate",
    "apply_cutoffs",
    "as_one_anndata",
    "asinh_transform",
    "assign_prelim",
    "barcode_key",
    "barcode_stats",
    "bead_baseline",
    "bead_channels",
    "bead_distance",
    "bead_gates",
    "channel_index",
    "compensate",
    "concat_samples",
    "debarcode",
    "density_image",
    "dna_channel",
    "estimate_cofactors",
    "estimate_cutoffs",
    "faded_colormap",
    "filter_events",
    "filter_log",
    "fluor_channels",
    "gate_beads",
    "gate_stats",
    "get_scale",
    "logicle_transform",
    "mass_channels",
    "normalise_beads",
    "plot_barcode_events",
    "plot_bead_gates",
    "plot_beads_over_time",
    "plot_biaxial",
    "plot_gate",
    "plot_yields",
    "polygon_mask",
    "read_controls",
    "read_fcs",
    "read_fcs_dir",
    "read_spillover",
    "record_filter",
    "remove_beads",
    "report",
    "spillover_from_controls",
    "subsample",
    "view",
    "write_spillover",
]


def __getattr__(name: str):
    # napari (and Qt) are heavy and optional: import them only on first use.
    if name in ("view", "CytoViewer", "Panel", "faded_colormap", "as_one_anndata"):
        from . import viewer as _viewer

        return getattr(_viewer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
