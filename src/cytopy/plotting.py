"""Static matplotlib figures: biaxial plots, gate overlays, and QC lines.

The napari viewer is for looking around; this is for the figures that go into
a report and stay put. Both draw the same thing the same way -- a smoothed 2-D
histogram from :func:`~cytopy.density_image` -- so a plot in a report matches
what was on screen when the gate was drawn.

Biaxial plots are coloured by event density by default. At a few million
events a per-point scatter is neither fast nor readable: the interesting
structure is where events pile up, and only a density does that honestly.
"""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np

from .density import Axes2D, density_image
from .scales import AsinhScale, LinearScale, LogicleScale, PretransformedScale, Scale
from .transforms import channel_index

__all__ = ["plot_biaxial", "plot_gate"]

#: Events drawn when a plot colours points individually rather than by density.
MAX_SCATTER = 50_000


def _matrix(adata: ad.AnnData, layer: str | None):
    return adata.X if layer is None else adata.layers[layer]


def axis_scale(adata: ad.AnnData, channel: str, layer: str | None) -> Scale:
    """How to label an axis for ``channel`` in ``layer``, in raw units if known.

    The same rule the viewer uses: a layer that recorded the transform which
    produced it gets ticks at decades of the original units, and anything else
    gets the stored values.

    Parameters
    ----------
    adata
        The AnnData being plotted.
    channel
        Channel on the axis.
    layer
        Layer being plotted, ``None`` for ``adata.X``.

    Returns
    -------
    Scale
        Maps plotted values to themselves; its inner transform, when it has
        one, places the raw-unit ticks.
    """
    if layer is None:
        return LinearScale()
    info = adata.uns.get("cytopy", {})
    j = channel_index(adata, channel)
    if layer == info.get("asinh_layer") and "cofactor" in adata.var:
        cofactor = float(adata.var["cofactor"].iloc[j])
        if np.isfinite(cofactor):
            return PretransformedScale(AsinhScale(cofactor=cofactor))
    if layer == info.get("logicle_layer"):
        params = info.get("logicle_params", {}).get(str(adata.var_names[j]))
        if params:
            return PretransformedScale(LogicleScale(**params))
    return LinearScale()


def _apply_ticks(axis, scale: Scale, lo: float, hi: float, which: str) -> None:
    ticks = scale.ticks(lo, hi)
    setter = axis.set_xticks if which == "x" else axis.set_yticks
    setter(ticks.major, ticks.labels)
    if len(ticks.minor):
        setter(ticks.minor, minor=True)


def plot_biaxial(
    adata: ad.AnnData,
    x: str,
    y: str,
    *,
    layer: str | None = None,
    cofactor: float | None = None,
    color_by: str = "density",
    subset: np.ndarray | str | None = None,
    gate: str | Sequence[str] | None = None,
    rectangles: Sequence[Sequence[float]] | None = None,
    bins: int = 512,
    smooth: float = 1.0,
    log: bool = True,
    cmap: str = "turbo",
    robust: bool = True,
    title: str | None = None,
    ax=None,
):
    """A two-channel plot, coloured by event density.

    Parameters
    ----------
    adata
        Events x channels AnnData.
    x, y
        Channels for the two axes. Any alias :func:`~cytopy.channel_index`
        accepts.
    layer
        Matrix to plot, ``None`` for ``adata.X``. Values are plotted as stored;
        the axes are labelled in raw units when the layer records its
        transform, exactly as in the viewer.
    cofactor
        Arcsinh the two channels for display only, and put the ticks back in
        raw units. Nothing is written to ``adata`` -- this is for plotting raw
        counts on a biexponential axis without materialising a layer for it,
        which on a run of millions of events is worth avoiding.
    color_by
        ``"density"`` (default) shades a smoothed 2-D histogram, which is the
        only thing that stays readable at millions of events. Otherwise the
        name of a boolean ``obs`` column -- a gate, or the bead mask -- which
        draws that group as points over a grey density of everything else,
        so a report can show what a step removed.
    subset
        Boolean mask, or the name of a boolean ``obs`` column, limiting the
        events drawn. ``None`` draws all of them.
    gate
        Gate name, or names, whose outline to draw over the plot. Gates
        recorded on channels other than ``x`` and ``y`` are skipped, since
        their outline would not mean anything here.
    rectangles
        Outlines to draw as ``(x_lo, x_hi, y_lo, y_hi)``, in plotted
        coordinates -- for a gate held as ranges rather than vertices, such as
        a bead gate.
    bins
        Resolution of the density, per axis.
    smooth
        Gaussian smoothing of the density, in bins.
    log
        Shade ``log1p(count)``, so rare populations stay visible.
    cmap
        Colormap for the density.
    robust
        Clip the axes to the 0.1-99.9th percentile, so a few extreme events
        cannot flatten the plot.
    title
        Axes title. Defaults to the event count.
    ax
        Existing axes to draw on; one is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Raises
    ------
    KeyError
        If ``color_by`` or ``subset`` names a column that is not in ``obs``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.0))

    matrix = _matrix(adata, layer)
    xi, yi = channel_index(adata, x), channel_index(adata, y)
    x_name, y_name = str(adata.var_names[xi]), str(adata.var_names[yi])

    mask = _resolve_mask(adata, subset)
    xv = np.asarray(matrix[:, xi], dtype=np.float64).ravel()[mask]
    yv = np.asarray(matrix[:, yi], dtype=np.float64).ravel()[mask]

    if cofactor is not None:
        xv = np.arcsinh(xv / cofactor)
        yv = np.arcsinh(yv / cofactor)
        x_scale = y_scale = PretransformedScale(AsinhScale(cofactor=float(cofactor)))
    else:
        x_scale = axis_scale(adata, x_name, layer)
        y_scale = axis_scale(adata, y_name, layer)
    quantiles = (0.001, 0.999) if robust else (0.0, 1.0)
    if xv.size == 0:
        x_lo, x_hi, y_lo, y_hi = 0.0, 1.0, 0.0, 1.0
    else:
        x_lo, x_hi = x_scale.limits(xv, quantiles=quantiles)
        y_lo, y_hi = y_scale.limits(yv, quantiles=quantiles)
    axes2d = Axes2D(x_lo, x_hi, y_lo, y_hi, bins=bins)

    highlight = None
    if color_by != "density":
        if color_by not in adata.obs:
            raise KeyError(f"no obs column {color_by!r} to colour by")
        highlight = np.asarray(adata.obs[color_by], dtype=bool)[mask]

    # The backdrop is every event in the subset; a highlight is drawn on top.
    image = density_image(
        xv if highlight is None else xv[~highlight],
        yv if highlight is None else yv[~highlight],
        axes2d,
        smooth=smooth,
        log=log,
    )
    ax.imshow(
        image,
        extent=(x_lo, x_hi, y_lo, y_hi),
        origin="upper",
        aspect="auto",
        cmap=cmap if highlight is None else "Greys",
        interpolation="nearest",
    )
    if highlight is not None:
        hx, hy = xv[highlight], yv[highlight]
        if hx.size > MAX_SCATTER:
            take = np.random.default_rng(0).choice(hx.size, MAX_SCATTER, replace=False)
            hx, hy = hx[take], hy[take]
        ax.scatter(hx, hy, s=1, c="#d62728", linewidths=0, alpha=0.5, label=color_by)
        ax.legend(loc="upper right", fontsize=7, frameon=False, markerscale=6)

    for name in _gate_names(gate):
        _draw_gate(ax, adata, name, x_name, y_name)
    for left, right, bottom, top in rectangles or []:
        ax.plot(
            [left, right, right, left, left],
            [bottom, bottom, top, top, bottom],
            color="#1f77b4",
            lw=1.2,
            ls="--",
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    _apply_ticks(ax, x_scale, x_lo, x_hi, "x")
    _apply_ticks(ax, y_scale, y_lo, y_hi, "y")
    ax.set_xlabel(x_name, fontsize=8)
    ax.set_ylabel(y_name, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title if title is not None else f"{int(xv.size):,} events", fontsize=9, loc="left")
    return ax


def _resolve_mask(adata: ad.AnnData, subset) -> np.ndarray:
    if subset is None:
        return np.ones(adata.n_obs, dtype=bool)
    if isinstance(subset, str):
        if subset not in adata.obs:
            raise KeyError(f"no obs column {subset!r} to subset on")
        return np.asarray(adata.obs[subset], dtype=bool)
    mask = np.asarray(subset, dtype=bool)
    if mask.shape[0] != adata.n_obs:
        raise ValueError(f"subset has {mask.shape[0]} entries, expected {adata.n_obs}")
    return mask


def _gate_names(gate) -> list[str]:
    if gate is None:
        return []
    return [gate] if isinstance(gate, str) else list(gate)


def _draw_gate(ax, adata: ad.AnnData, name: str, x_name: str, y_name: str) -> None:
    """Outline a recorded gate, if it was drawn on these two channels."""
    record = adata.uns.get("cytopy", {}).get("gates", {}).get(name)
    if record is None:
        return
    if str(record.get("x", "")) != x_name or str(record.get("y", "")) != y_name:
        return
    vertices = record.get("vertices")
    if vertices is None:
        return
    for shape in vertices:
        shape = np.asarray(shape, dtype=float)
        if shape.shape[0] < 2:
            continue
        closed = np.vstack([shape, shape[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="#1f77b4", lw=1.2, ls="--")
    ax.annotate(
        f"{name}: {int(record.get('n', 0)):,}",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=7,
        color="#1f77b4",
    )


def plot_gate(adata: ad.AnnData, name: str, *, layer: str | None = None, ax=None, **kwargs):
    """A gate in the plane it was drawn in, with the events it kept picked out.

    Reads the channels, layer and outline back from
    ``adata.uns['cytopy']['gates'][name]``, so a gate drawn in the viewer
    weeks ago redraws exactly where it was.

    Parameters
    ----------
    adata
        AnnData carrying the gate as a boolean ``obs`` column.
    name
        The gate to draw.
    layer
        Override the layer the gate was recorded on.
    ax
        Existing axes to draw on.
    **kwargs
        Passed to :func:`plot_biaxial`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Raises
    ------
    KeyError
        If the gate was never recorded, or did not record its channels.
    """
    gates = adata.uns.get("cytopy", {}).get("gates", {})
    if name not in gates:
        raise KeyError(f"no gate {name!r}; recorded gates are {sorted(gates)}")
    record = gates[name]
    if "x" not in record or "y" not in record:
        raise KeyError(f"gate {name!r} did not record which channels it was drawn on")
    if layer is None:
        stored = str(record.get("layer", "X"))
        layer = None if stored in ("X", "") else stored
    kwargs.setdefault("color_by", name)
    kwargs.setdefault("subset", record.get("parent") or None)
    kwargs.setdefault("title", f"{name} ({int(record.get('n', 0)):,} events)")
    return plot_biaxial(adata, record["x"], record["y"], layer=layer, gate=name, ax=ax, **kwargs)
