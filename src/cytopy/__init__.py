"""cytopy: cytometry analysis on AnnData, gated interactively in napari."""

from __future__ import annotations

from .density import Axes2D, density_curve, density_image
from .filters import filter_events, filter_log, record_filter
from .gating import (
    GateRecord,
    add_gate,
    ellipse_mask,
    gate_children,
    gate_mask,
    gate_order,
    gate_record,
    gate_stats,
    polygon_mask,
    recompute_gates,
    rectangle_to_polygon,
    shapes_mask,
)
from .io import concat_samples, read_fcs, read_fcs_dir, split_samples
from .plotting import plot_biaxial, plot_compensation, plot_gate
from .report import gating_pdf, report
from .scales import (
    AsinhScale,
    LinearScale,
    LogicleScale,
    LogScale,
    PretransformedScale,
    Scale,
    get_scale,
)
from .spillover import (
    compensate,
    compensation_residuals,
    compute_spillover_matrix,
    read_controls,
    read_spillover,
    subset_controls,
    write_spillover,
)
from .transforms import (
    asinh_transform,
    channel_index,
    estimate_cofactors,
    fluor_channels,
    logicle_transform,
    subsample,
)

__version__ = "0.0.1-alpha"

__all__ = [
    "AsinhScale",
    "Axes2D",
    "CytoViewer",
    "GateRecord",
    "LinearScale",
    "LogScale",
    "LogicleScale",
    "Panel",
    "PretransformedScale",
    "Scale",
    "add_gate",
    "as_one_anndata",
    "asinh_transform",
    "channel_index",
    "compensate",
    "compensation_residuals",
    "compute_spillover_matrix",
    "concat_samples",
    "current_viewer",
    "density_curve",
    "density_image",
    "ellipse_mask",
    "estimate_cofactors",
    "faded_colormap",
    "filter_events",
    "filter_log",
    "fluor_channels",
    "gate_children",
    "gate_mask",
    "gate_order",
    "gate_record",
    "gate_stats",
    "gating_pdf",
    "get_scale",
    "logicle_transform",
    "open_napari",
    "plot_biaxial",
    "plot_compensation",
    "plot_gate",
    "polygon_mask",
    "read_controls",
    "read_fcs",
    "read_fcs_dir",
    "read_spillover",
    "recompute_gates",
    "record_filter",
    "rectangle_to_polygon",
    "report",
    "shapes_mask",
    "split_samples",
    "subsample",
    "subset_controls",
    "write_spillover",
]


#: Names served from :mod:`cytopy.viewer`, which pulls in napari and Qt. Listed
#: here rather than read off ``viewer.__all__`` because importing the module to
#: find out would defeat the point; ``tests/test_api.py`` checks the two agree.
_LAZY = (
    "CytoViewer",
    "Panel",
    "as_one_anndata",
    "current_viewer",
    "faded_colormap",
    "open_napari",
)


def __getattr__(name: str):
    # napari (and Qt) are heavy and optional: import them only on first use.
    if name in _LAZY:
        from . import viewer as _viewer

        return getattr(_viewer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
