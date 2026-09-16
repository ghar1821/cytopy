"""napari front-end: 2-channel density plots with polygon gating.

The viewer never transforms data. It plots the values of the layer you point
it at, exactly as they are stored, so what you see is the result of the
transform *you* ran (:func:`cytopy.asinh_transform`,
:func:`cytopy.logicle_transform`, ...). The only thing it infers is how to
*label* the axes: when a layer records the transform that produced it, the
ticks are drawn as decades of the original units, which is what makes an
arcsinh or logicle layer read as a biexponential plot.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .beads import (
    BEAD_COFACTOR,
    BEAD_PANELS,
    bead_channels,
    bead_gates,
    dna_channel,
    gate_beads,
    mass_channels,
    normalise_beads,
)
from .density import Axes2D, density_curve, density_image
from .gating import add_gate, shapes_mask
from .scales import AsinhScale, LinearScale, LogicleScale, PretransformedScale, Scale
from .transforms import asinh_transform, channel_index

__all__ = ["CytoViewer", "view"]

TICK_CHOICES = ["auto", "linear"]
COLORMAPS = ["turbo", "viridis", "magma", "inferno", "gray", "plasma"]

#: Canvas background. White by default: a density plot reads as a flow plot on
#: white, and napari's own canvas is a dark blue-grey that most cytometry
#: software does not use. Any napari colour works -- a name, or "#rrggbb".
BACKGROUNDS = ["white", "black", "#262930", "#f5f5f5"]
DEFAULT_BACKGROUND = "white"

#: How much of the colormap fades out at the bottom, so that empty bins show
#: the background instead of the colormap's darkest colour.
_FADE = 0.02

_TEXT_STYLE = {"size": 9, "anchor": "center"}

_PANEL = "panel"
_COMPARE = "density (compare)"
_COLORBAR = "colour bar"
_GRID = "axes"
_LABELS = "axis labels"
_GATES = "gates"
_BEAD_GATE = "bead gate"

#: What the canvas draws. ``"density"`` is the two-channel plot;
#: ``"histogram"`` is one smoothed distribution of the x channel per sample.
PLOT_KINDS = ["density", "histogram"]

#: Curve colours for the histogram, one per sample, reused beyond eight.
CURVE_COLOURS = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#b279a2",
    "#9d755d",
    "#eeca3b",
    "#72b7b2",
]

#: How the comparison panel is shown against the main one.
COMPARE_MODES = ["side by side", "overlay"]
_OFF = "<off>"

#: Comparison against another pair of channels rather than another population.
#: The one that makes a second plot out of a single file.
_CHANNELS = "another channel pair"

#: Colormaps for the two densities in overlay mode. Single-hue on purpose:
#: additive blending only reads as "one, the other, or both" when each side
#: has a hue of its own, which a multi-hue map like turbo cannot give it. The
#: **colormap** setting governs the single-panel and side-by-side views.
OVERLAY_COLORMAPS = ("green", "magenta")

#: Gap between panels, and the colour bar's width, as fractions of ``bins``.
_PANEL_GAP = 0.12
_BAR_WIDTH = 0.035

#: Layer the bead panel plots: arcsinh with the mass cytometry cofactor, which
#: is the space premessa's bead gates are defined in.
BEAD_LAYER = "beadscale"

#: Events the bead panel estimates its live count from while a gate is being
#: dragged. Applying the gate always uses every event.
BEAD_PREVIEW_EVENTS = 250_000


class Panel:
    """One plot on the canvas: what it shows, where it sits, and its layers.

    A panel owns the image (or the curves) it draws into, so it appears in
    napari's own layer list and can be hidden or reordered there. Its frame and
    labels are drawn from :attr:`row` and :attr:`col`, so the box travels with
    the plot when it is moved.

    Parameters
    ----------
    viewer
        The napari viewer to add the layers to.
    index
        Number used to name the layers; unique within a window.
    kind
        ``"density"`` for the two-channel plot, ``"histogram"`` for one
        smoothed distribution of ``x`` per sample.
    layer
        Matrix to plot: ``"X"`` or a key of ``adata.layers``.
    x, y
        Channels on each axis. ``y`` is unused by a histogram.
    sample
        Sample to show, or ``"<all>"``.
    parent
        Gate to restrict to, or ``"<none>"``.
    row, col
        Position of the panel's top-left corner, in bin units.
    colormap
        Colormap for the density image.
    """

    def __init__(
        self,
        viewer,
        index: int,
        *,
        kind: str = "density",
        layer: str = "X",
        x: str = "",
        y: str = "",
        sample: str = "<all>",
        parent: str = "<none>",
        row: float = 0.0,
        col: float = 0.0,
        colormap: str = "turbo",
    ):
        self.index = int(index)
        self.kind = kind
        self.layer = layer
        self.x = x
        self.y = y
        self.sample = sample
        self.parent = parent
        self.row = float(row)
        self.col = float(col)

        self.axes: Axes2D | None = None
        self.x_scale: Scale | None = None
        self.y_scale: Scale | None = None
        self.peak = 0.0
        self.legend: list[tuple[float, float, str]] = []

        self.image = viewer.add_image(
            np.zeros((2, 2), dtype=np.float32),
            name=f"{_PANEL} {index}",
            colormap=faded_colormap(colormap),
            interpolation2d="nearest",
            blending="translucent_no_depth",
        )
        self.curves = viewer.add_shapes(
            name=f"{_PANEL} {index} curves", shape_type="path", edge_width=2, visible=False
        )
        self.curves.editable = False

    # ------------------------------------------------------------------ state
    @property
    def histogram(self) -> bool:
        """Whether this panel draws distributions rather than a density."""
        return self.kind == "histogram"

    @property
    def title(self) -> str:
        """Short description, written above the panel when there is more than one."""
        where = "" if self.sample == "<all>" else f"  {self.sample}"
        if self.histogram:
            return f"{self.x}{where}"
        return f"{self.x} × {self.y}{where}"

    def settings(self) -> dict:
        """The panel's settings, as keyword arguments for a new one."""
        return {
            "kind": self.kind,
            "layer": self.layer,
            "x": self.x,
            "y": self.y,
            "sample": self.sample,
            "parent": self.parent,
        }

    def contains(self, row: float, col: float, bins: int) -> bool:
        """Whether a canvas point falls inside this panel's box.

        Parameters
        ----------
        row, col
            Canvas coordinates.
        bins
            Current panel size, in pixels.

        Returns
        -------
        bool
        """
        return self.row - 0.5 <= row <= self.row + bins and self.col - 0.5 <= col <= self.col + bins

    def remove_from(self, viewer) -> None:
        """Take this panel's layers out of the viewer.

        Parameters
        ----------
        viewer
            The napari viewer holding them.
        """
        for layer in (self.image, self.curves):
            if layer in viewer.layers:
                viewer.layers.remove(layer)


class CytoViewer:
    """A napari window showing a 2-channel density plot of an AnnData.

    Parameters
    ----------
    adata
        What to plot. One events x channels AnnData, e.g. from
        :func:`cytopy.read_fcs`; a path to an FCS file, an ``.h5ad`` or a
        directory; or several of either as a list or as a
        ``{name: object}`` mapping. Several are concatenated on the channels
        they share and become entries in the **sample** selector.
    names
        Names for the samples, overriding whatever they carry. One per object.
    layer
        Which matrix to plot: ``None`` for ``adata.X``, otherwise a key of
        ``adata.layers`` (``"comp"``, ``"asinh"``, ``"logicle"``, ...). Its
        values are plotted as-is.
    x, y
        Channels to put on each axis initially. Each may be a ``var_name``, a
        ``$PnN`` detector (``"FITC-A"``) or a ``$PnS`` marker (``"CD3"``).
        Default to the first two channels in the file.
    ticks
        ``"auto"`` labels the axes in raw units when the layer knows which
        transform produced it, and falls back to the stored values otherwise.
        ``"linear"`` always labels the stored values.
    bins
        Resolution of the density image, per axis.
    robust
        Clip the axis range to the 0.1-99.9th percentile, so a handful of
        extreme events (common after compensation) cannot flatten the plot.
        Events outside the range are not drawn and fall outside any gate.
    colormap
        Initial colormap for the density image; one of :data:`COLORMAPS`.
    smooth
        Gaussian smoothing of the density, in bins. ``0`` shows raw counts.
    background
        Canvas colour behind the plot, ``"white"`` by default. Any napari
        colour: a name, or ``"#rrggbb"``. Empty bins are transparent, so this
        is what shows through them, and the axis text, ticks and gate outlines
        switch between light and dark to stay legible against it.
    viewer
        An existing ``napari.Viewer`` to build into. A new one is created when
        omitted; pass one to embed the plot in a window you already have.
    title
        Window title, used only when creating a viewer.

    Notes
    -----
    The plot is a 2-D histogram drawn as a napari image layer, so canvas
    coordinates are histogram pixels. :attr:`axes` converts between those and
    the values being plotted.

    Each plot is a :class:`Panel`, owning the napari layer it draws into. That
    layer is rewritten on every redraw, so duplicating it in napari gives a
    frozen snapshot rather than a second plot -- and once the axes move it is a
    snapshot in a coordinate frame that no longer applies. Add a panel instead:
    it keeps its own settings, shares the intensity scale with the others, and
    redraws with them.
    """

    def __init__(
        self,
        adata,
        *,
        names: Sequence[str] | None = None,
        layer: str | None = None,
        x: str | None = None,
        y: str | None = None,
        ticks: str = "auto",
        bins: int = 512,
        colormap: str = "turbo",
        smooth: float = 1.0,
        robust: bool = True,
        background: str = DEFAULT_BACKGROUND,
        viewer=None,
        title: str = "cytopy",
    ):
        """Build the layers and the control panel, then draw. See the class docstring."""
        import napari

        adata = as_one_anndata(adata, names=names)
        self.adata = adata
        self.bins = int(bins)
        self.smooth = float(smooth)
        self.panels: list[Panel] = []
        self.active = 0
        self._counter = 0
        self._layout = None
        self._derive_colours(background)
        self._updating = False

        channels = list(adata.var_names)
        if not channels:
            raise ValueError("adata has no channels")
        # Accept a marker ("CD3") or a detector ("FITC-A") as well as a var_name.
        x = channels[channel_index(adata, x)] if x else channels[0]
        y = channels[channel_index(adata, y)] if y else channels[min(1, len(channels) - 1)]

        self.viewer = viewer if viewer is not None else napari.Viewer(title=title)
        _hide_overlays(self.viewer)

        self._init_layers(colormap)
        self._init_widget(layer, x, y, ticks, colormap, robust)
        self._counter = 1
        self.panels.append(
            Panel(
                self.viewer,
                1,
                x=x,
                y=y,
                layer="X" if layer is None else layer,
                colormap=colormap,
            )
        )
        self._bind()
        self.set_background(self.background)
        self.viewer.mouse_drag_callbacks.append(self._on_mouse)
        self.refresh()
        self.viewer.reset_view()

    # ------------------------------------------------------------------ setup
    def _init_layers(self, colormap: str) -> None:
        """Build the layers every panel shares; panels bring their own."""
        self.colorbar = self.viewer.add_image(
            np.zeros((2, 1), dtype=np.float32),
            name=_COLORBAR,
            colormap=faded_colormap(colormap),
            interpolation2d="nearest",
            blending="translucent_no_depth",
        )
        self.colorbar.editable = False
        self.grid = self.viewer.add_shapes(
            name=_GRID, shape_type="line", edge_color=self._grid_colour, edge_width=1, opacity=0.6
        )
        self.grid.editable = False
        self.labels = self.viewer.add_points(
            np.empty((0, 2)),
            name=_LABELS,
            size=1,
            face_color="transparent",
            border_color="transparent",
            text={**_TEXT_STYLE, "string": [], "color": self._foreground},
        )
        self.labels.editable = False
        self.gates = self.viewer.add_shapes(
            name=_GATES, face_color="#ffcc0022", edge_color="#ffcc00", edge_width=2
        )
        # Read-only outline of the current bead gate, redrawn from the spin
        # boxes; the editable shapes stay on the gates layer.
        self.bead_gate = self.viewer.add_shapes(
            name=_BEAD_GATE, face_color="transparent", edge_color="#4da6ff", edge_width=2
        )
        self.bead_gate.editable = False
        self.bead_gate.visible = False
        self.viewer.layers.selection = {self.gates}

    # ------------------------------------------------------------------ panels
    @property
    def panel(self) -> Panel:
        """The panel the settings currently apply to."""
        return self.panels[self.active]

    @property
    def axes(self) -> Axes2D | None:
        """Axes of the active panel, which is what gates are drawn against."""
        return self.panel.axes

    @property
    def density(self):
        """The active panel's image layer."""
        return self.panel.image

    @property
    def curves(self):
        """The active panel's curve layer."""
        return self.panel.curves

    @property
    def x_scale(self) -> Scale | None:
        """Axis scale of the active panel's x channel."""
        return self.panel.x_scale

    @property
    def y_scale(self) -> Scale | None:
        """Axis scale of the active panel's y channel."""
        return self.panel.y_scale

    def add_panel(self, kind: str = "density", **settings) -> Panel:
        """Add a plot to the canvas and make it the active one.

        Parameters
        ----------
        kind
            ``"density"`` or ``"histogram"``.
        **settings
            Overrides for the new panel: ``layer``, ``x``, ``y``, ``sample``,
            ``parent``. Anything not given is copied from the active panel, so
            adding one gives you the same view to then change.

        Returns
        -------
        Panel
            The panel that was added.
        """
        base = self.panel.settings() if self.panels else {"x": "", "y": ""}
        base.update(settings)
        base["kind"] = kind
        self._counter += 1
        panel = Panel(self.viewer, self._counter, colormap=self.w_cmap.value, **base)
        self.panels.append(panel)
        self.arrange()
        self.select_panel(len(self.panels) - 1)
        return panel

    def remove_panel(self, index: int | None = None) -> None:
        """Take a panel off the canvas.

        Parameters
        ----------
        index
            Which panel; the active one by default.

        Raises
        ------
        ValueError
            If it is the only panel left -- a viewer with no plot in it is not
            a useful thing to be able to make by accident.
        """
        if len(self.panels) == 1:
            raise ValueError("a viewer needs at least one panel")
        index = self.active if index is None else int(index)
        self.panels.pop(index).remove_from(self.viewer)
        self.arrange()
        self.select_panel(min(index, len(self.panels) - 1))

    def select_panel(self, index: int) -> None:
        """Make a panel active, so the settings on the right apply to it.

        Parameters
        ----------
        index
            Position in :attr:`panels`.
        """
        self.active = int(index) % len(self.panels)
        self._bind()
        self.refresh()

    def arrange(self, columns: int | None = None) -> None:
        """Tile the panels into a grid.

        Parameters
        ----------
        columns
            Panels per row. Defaults to a roughly square grid.
        """
        count = len(self.panels)
        columns = columns or max(int(np.ceil(np.sqrt(count))), 1)
        step = self.bins * (1.0 + _PANEL_GAP)
        for i, panel in enumerate(self.panels):
            panel.row = (i // columns) * step
            panel.col = (i % columns) * step

    def _init_widget(self, layer, x, y, ticks, colormap, robust) -> None:
        from magicgui.widgets import (
            CheckBox,
            ComboBox,
            Container,
            FloatSpinBox,
            Label,
            LineEdit,
            PushButton,
            SpinBox,
        )

        names = list(self.adata.var_names)
        layer_choices = ["X"] + [k for k in self.adata.layers if k is not None]
        layer_value = "X" if layer is None else layer
        if layer_value not in layer_choices:
            raise KeyError(f"layer {layer!r} not in {layer_choices}")

        # --- which panel the settings below apply to --------------------
        self.w_panel = ComboBox(label="panel", choices=("1",), value="1")
        self.w_add_density = PushButton(text="+ density")
        self.w_add_histogram = PushButton(text="+ histogram")
        self.w_remove = PushButton(text="remove panel")
        self.w_arrange = PushButton(text="arrange")
        self.w_move = CheckBox(label="move panels (drag)", value=False)

        # --- per panel --------------------------------------------------
        self.w_plot = ComboBox(label="plot", choices=PLOT_KINDS, value=PLOT_KINDS[0])
        self.w_layer = ComboBox(label="data", choices=layer_choices, value=layer_value)
        self.w_x = ComboBox(label="x channel", choices=names, value=x)
        self.w_y = ComboBox(label="y channel", choices=names, value=y)
        self.w_swap = PushButton(text="swap x / y")
        self.w_ticks = ComboBox(label="axis ticks", choices=TICK_CHOICES, value=ticks)
        self.w_transform = Label(value="")
        samples = ["<all>"]
        if "sample" in self.adata.obs:
            samples += [str(s) for s in self.adata.obs["sample"].astype(str).unique()]
        self.w_sample = ComboBox(label="sample", choices=samples, value="<all>")
        self.w_parent = ComboBox(label="parent gate", choices=self._gate_choices(), value="<none>")

        # --- global -----------------------------------------------------
        self.w_bins = SpinBox(label="bins", value=self.bins, min=64, max=2048, step=64)
        self.w_smooth = FloatSpinBox(label="smoothing", value=self.smooth, min=0, max=10, step=0.5)
        self.w_log = CheckBox(label="log counts", value=True)
        self.w_robust = CheckBox(label="robust limits", value=bool(robust))
        self.w_cmap = ComboBox(label="colormap", choices=COLORMAPS, value=colormap)
        self.w_background = ComboBox(
            label="background",
            choices=sorted({*BACKGROUNDS, self.background}),
            value=self.background,
        )
        self.w_colorbar = CheckBox(label="colour bar", value=True)
        self.w_peak = CheckBox(label="scale each curve to its peak", value=False)
        self.w_lock = CheckBox(label="same axes across samples", value=len(samples) > 2)

        self.w_gate_name = LineEdit(label="gate name", value="gate1")
        self.w_gate_apply = PushButton(text="apply gate from shapes")
        self.w_gate_clear = PushButton(text="clear shapes")
        self.w_status = Label(value="")

        panels = Container(
            widgets=[
                self.w_panel,
                self.w_add_density,
                self.w_add_histogram,
                self.w_remove,
                self.w_arrange,
                self.w_move,
            ],
            label="panels",
        )
        plot = Container(
            widgets=[
                self.w_plot,
                self.w_layer,
                self.w_x,
                self.w_y,
                self.w_swap,
                self.w_sample,
                self.w_parent,
                self.w_ticks,
                self.w_transform,
            ],
            label="this panel",
        )
        style = Container(
            widgets=[
                self.w_bins,
                self.w_smooth,
                self.w_log,
                self.w_robust,
                self.w_lock,
                self.w_cmap,
                self.w_background,
                self.w_colorbar,
                self.w_peak,
            ],
            label="display (all panels)",
        )
        gate = Container(
            widgets=[self.w_gate_name, self.w_gate_apply, self.w_gate_clear], label="gating"
        )
        sections = [panels, plot, style, gate]
        beads = self._init_bead_widget()
        if beads is not None:
            sections.append(beads)
        self.widget = Container(widgets=[*sections, self.w_status])

        # Per-panel settings write to the active panel, then redraw.
        for w in (self.w_plot, self.w_layer, self.w_x, self.w_y, self.w_sample, self.w_parent):
            w.changed.connect(self._panel_changed)
        for w in (
            self.w_ticks,
            self.w_bins,
            self.w_smooth,
            self.w_log,
            self.w_robust,
            self.w_lock,
            self.w_colorbar,
            self.w_peak,
        ):
            w.changed.connect(self._on_change)
        self.w_cmap.changed.connect(self._on_change)
        self.w_background.changed.connect(lambda e: self.set_background(self.w_background.value))
        self.w_swap.changed.connect(self._swap)
        self.w_gate_apply.changed.connect(self._apply_gate)
        self.w_gate_clear.changed.connect(lambda e: setattr(self.gates, "data", []))
        self.w_panel.changed.connect(lambda e: self.select_panel(int(str(self.w_panel.value)) - 1))
        self.w_add_density.changed.connect(lambda e: self.add_panel("density"))
        self.w_add_histogram.changed.connect(lambda e: self.add_panel("histogram"))
        self.w_remove.changed.connect(self._remove_clicked)
        self.w_arrange.changed.connect(lambda e: (self.arrange(), self.refresh()))

        self.viewer.window.add_dock_widget(self.widget, area="right", name="cytopy")

    # ------------------------------------------------------------- selection
    def _bind(self) -> None:
        """Point the per-panel widgets at the active panel."""
        panel = self.panel
        self._updating = True
        try:
            self.w_panel.choices = tuple(str(p.index) for p in self.panels)
            self.w_panel.value = str(panel.index)
            self.w_plot.value = panel.kind
            self.w_layer.value = panel.layer
            self.w_x.value = panel.x
            self.w_y.value = panel.y
            self.w_sample.value = panel.sample
            self.w_parent.value = panel.parent
        finally:
            self._updating = False

    def _panel_changed(self, *_) -> None:
        """Copy the per-panel widgets onto the active panel."""
        if self._updating:
            return
        panel = self.panel
        panel.kind = str(self.w_plot.value)
        panel.layer = str(self.w_layer.value)
        panel.x = str(self.w_x.value)
        panel.y = str(self.w_y.value)
        panel.sample = str(self.w_sample.value)
        panel.parent = str(self.w_parent.value)
        self.refresh()

    def _remove_clicked(self, *_) -> None:
        try:
            self.remove_panel()
        except ValueError as exc:
            self.w_status.value = str(exc)

    def _panel_at(self, row: float, col: float) -> int | None:
        """Index of the panel under a canvas point, topmost first."""
        for index in reversed(range(len(self.panels))):
            if self.panels[index].contains(row, col, self.bins):
                return index
        return None

    def _on_mouse(self, viewer, event):
        """Select the panel under the cursor, and drag it when asked to."""
        position = getattr(event, "position", None)
        if position is None or len(position) < 2:
            return
        index = self._panel_at(position[-2], position[-1])
        if index is None:
            return
        if index != self.active:
            self.select_panel(index)
        if not self.w_move.value:
            return

        panel = self.panels[index]
        start = (position[-2], position[-1], panel.row, panel.col)
        yield
        while event.type == "mouse_move":
            here = event.position
            panel.row = start[2] + (here[-2] - start[0])
            panel.col = start[3] + (here[-1] - start[1])
            panel.image.translate = (panel.row, panel.col)
            panel.curves.translate = (panel.row, panel.col)
            yield
        self.refresh()

    # ------------------------------------------------------------------ beads
    def _init_bead_widget(self):
        """Build the bead-normalisation panel, or return None if there are no beads.

        premessa's workflow, in the plot that is already on screen: put a bead
        channel against DNA, drag the rectangle until it holds the beads and
        nothing else, apply, normalise. Nothing happens to the data until you
        ask: the first click creates the ``arcsinh(x / 5)`` layer the gates are
        defined on and switches the plot to it.
        """
        from magicgui.widgets import ComboBox, Container, FloatSpinBox, Label, PushButton

        try:
            channels = bead_channels(self.adata, "fluidigm")
            self.dna = dna_channel(self.adata)
        except KeyError:
            self.bead_gates = None
            return None  # not mass cytometry, or not a panel with bead channels

        self.bead_gates = bead_gates(self.adata, "fluidigm")
        self._bead_sample: ad.AnnData | None = None
        self.w_bead_set = ComboBox(label="bead set", choices=sorted(BEAD_PANELS), value="fluidigm")
        self.w_bead_channel = ComboBox(label="bead channel", choices=channels, value=channels[0])
        spin = {"min": -20.0, "max": 20.0, "step": 0.1}
        self.w_bead_xlo = FloatSpinBox(label="bead min", **spin)
        self.w_bead_xhi = FloatSpinBox(label="bead max", **spin)
        self.w_bead_ylo = FloatSpinBox(label="DNA min", **spin)
        self.w_bead_yhi = FloatSpinBox(label="DNA max", **spin)
        self.w_bead_show = PushButton(text="plot bead channel vs DNA")
        self.w_bead_from_shape = PushButton(text="gate from drawn rectangle")
        self.w_bead_apply = PushButton(text="apply bead gate")
        self.w_bead_norm = PushButton(text="normalise")
        self.w_bead_status = Label(value="")

        self.w_bead_set.changed.connect(self._bead_set_changed)
        self.w_bead_channel.changed.connect(self._bead_channel_changed)
        for w in (self.w_bead_xlo, self.w_bead_xhi, self.w_bead_ylo, self.w_bead_yhi):
            w.changed.connect(self._bead_gate_edited)
        self.w_bead_show.changed.connect(self._bead_channel_changed)
        self.w_bead_from_shape.changed.connect(self._bead_gate_from_shape)
        self.w_bead_apply.changed.connect(self._apply_bead_gate)
        self.w_bead_norm.changed.connect(self._normalise_beads)

        self._load_bead_gate()
        return Container(
            widgets=[
                self.w_bead_set,
                self.w_bead_channel,
                self.w_bead_show,
                self.w_bead_xlo,
                self.w_bead_xhi,
                self.w_bead_ylo,
                self.w_bead_yhi,
                self.w_bead_from_shape,
                self.w_bead_apply,
                self.w_bead_norm,
                self.w_bead_status,
            ],
            label="beads",
        )

    def _ensure_bead_layer(self) -> None:
        """Plot arcsinh(counts / 5), the space the bead gates are defined in."""
        if BEAD_LAYER not in self.adata.layers:
            asinh_transform(
                self.adata,
                BEAD_COFACTOR,
                channels=[*mass_channels(self.adata), self.dna],
                key_added=BEAD_LAYER,
            )
        self._updating = True
        try:
            if BEAD_LAYER not in list(self.w_layer.choices):
                self.w_layer.choices = ["X"] + [k for k in self.adata.layers if k is not None]
            self.w_layer.value = BEAD_LAYER
            self.w_x.value = self.w_bead_channel.value
            self.w_y.value = self.dna
            self.w_ticks.value = "linear"
        finally:
            self._updating = False

    def _load_bead_gate(self, *_) -> None:
        """Push the stored gate for the current bead channel into the spin boxes."""
        gate = self.bead_gates[str(self.w_bead_channel.value)]
        self._updating = True
        try:
            self.w_bead_xlo.value, self.w_bead_xhi.value = gate["x"]
            self.w_bead_ylo.value, self.w_bead_yhi.value = gate["y"]
        finally:
            self._updating = False

    def _bead_set_changed(self, *_) -> None:
        try:
            channels = bead_channels(self.adata, str(self.w_bead_set.value))
        except KeyError as exc:
            self.w_bead_status.value = str(exc).split(";")[0]
            return
        self.bead_gates = bead_gates(self.adata, str(self.w_bead_set.value))
        self._bead_sample = None
        self._updating = True
        try:
            self.w_bead_channel.choices = channels
            self.w_bead_channel.value = channels[0]
        finally:
            self._updating = False
        self._bead_channel_changed()

    def _bead_channel_changed(self, *_) -> None:
        if self._updating:
            return
        self._load_bead_gate()
        self._ensure_bead_layer()
        self.refresh()

    def _bead_gate_edited(self, *_) -> None:
        if self._updating:
            return
        self.bead_gates[str(self.w_bead_channel.value)] = {
            "x": (float(self.w_bead_xlo.value), float(self.w_bead_xhi.value)),
            "y": (float(self.w_bead_ylo.value), float(self.w_bead_yhi.value)),
        }
        self._draw_bead_gate()
        self._count_beads()

    def _bead_gate_from_shape(self, *_) -> None:
        """Adopt the bounding box of a rectangle drawn on the gates layer."""
        if self.axes is None or not len(self.gates.data):
            self.w_bead_status.value = "draw a rectangle on the canvas first"
            return
        verts = self.axes.to_display(np.asarray(self.gates.data[-1], dtype=float))
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        self._updating = True
        try:
            self.w_bead_xlo.value, self.w_bead_xhi.value = float(lo[0]), float(hi[0])
            self.w_bead_ylo.value, self.w_bead_yhi.value = float(lo[1]), float(hi[1])
        finally:
            self._updating = False
        self._bead_gate_edited()

    def _draw_bead_gate(self) -> None:
        """Outline the current channel's gate on the canvas."""
        gate = self.bead_gates.get(str(self.w_x.value))
        if self.axes is None or gate is None or self.w_layer.value != BEAD_LAYER:
            self.bead_gate.data = []
            self.bead_gate.visible = False
            return
        (x_lo, x_hi), (y_lo, y_hi) = gate["x"], gate["y"]
        corners = self.axes.to_pixels(
            np.array([x_lo, x_hi, x_hi, x_lo]), np.array([y_lo, y_lo, y_hi, y_hi])
        )
        self.bead_gate.data = []
        self.bead_gate.add_rectangles([corners])
        self.bead_gate.visible = True

    def bead_mask(self, adata: ad.AnnData | None = None) -> np.ndarray:
        """Events inside the current bead gate, across every bead channel.

        Parameters
        ----------
        adata
            What to gate. Defaults to the viewer's own data; the panel passes
            a subsample here to keep the live count cheap while a gate is
            being dragged.

        Returns
        -------
        ndarray
            Boolean mask over the events of ``adata``.
        """
        return gate_beads(
            adata if adata is not None else self.adata,
            beads=list(self.bead_gates),
            dna=self.dna,
            gates=self.bead_gates,
            key_added=None,
        )

    def _bead_preview(self) -> ad.AnnData:
        """A fixed subsample the live count is estimated from.

        Dragging a gate edge re-counts on every step, and at a few million
        events an exact count makes the spin boxes lag. The subsample is drawn
        once and reused, so the number moves smoothly as the gate moves and is
        never off by more than sampling error; *apply* uses every event.
        """
        if self._bead_sample is None:
            if self.adata.n_obs <= BEAD_PREVIEW_EVENTS:
                self._bead_sample = self.adata
            else:
                rng = np.random.default_rng(0)
                take = rng.choice(self.adata.n_obs, BEAD_PREVIEW_EVENTS, replace=False)
                self._bead_sample = self.adata[np.sort(take)].copy()
        return self._bead_sample

    def _count_beads(self) -> None:
        preview = self._bead_preview()
        try:
            n = int(self.bead_mask(preview).sum())
        except ValueError:
            self.w_bead_status.value = "gate holds no events"
            return
        fraction = n / preview.n_obs
        estimate = "" if preview is self.adata else " est."
        self.w_bead_status.value = (
            f"{round(fraction * self.adata.n_obs):,} beads{estimate} ({100 * fraction:.2f}%)"
        )

    def _apply_bead_gate(self, *_) -> None:
        try:
            mask = gate_beads(
                self.adata,
                beads=list(self.bead_gates),
                dna=self.dna,
                gates=self.bead_gates,
                key_added="bead",
            )
        except ValueError as exc:
            self.w_bead_status.value = str(exc).split(";")[0]
            return
        self._updating = True
        try:
            self.w_parent.choices = self._gate_choices()
        finally:
            self._updating = False
        self.w_bead_status.value = f"obs['bead']: {int(mask.sum()):,} events"

    def _normalise_beads(self, *_) -> None:
        if "bead" not in self.adata.obs:
            self._apply_bead_gate()
        if "bead" not in self.adata.obs:
            return
        try:
            normalise_beads(self.adata, beads=list(self.bead_gates), dna=self.dna)
        except (ValueError, KeyError) as exc:
            self.w_bead_status.value = str(exc).split(";")[0]
            return
        self.adata.layers.pop(BEAD_LAYER, None)
        self._updating = True
        try:
            self.w_layer.choices = ["X"] + [k for k in self.adata.layers if k is not None]
            self.w_layer.value = "normalised"
        finally:
            self._updating = False
        slopes = self.adata.obs["bead_slope"]
        self.w_bead_status.value = (
            f"normalised -> layers['normalised']; slope {slopes.min():.3f}-{slopes.max():.3f}"
        )
        self.refresh()

    # ------------------------------------------------------------ appearance
    def _derive_colours(self, colour: str) -> str:
        """Store the background and the decoration colours that suit it.

        Returns the gate colour, which only ``set_background`` needs.
        """
        from napari.utils.colormaps.standardize_color import transform_color

        transform_color(colour)  # fail here, not three layers later
        self.background = str(colour)
        light = _is_light(colour)
        self._foreground = "#1e1e21" if light else "#f0f1f2"
        self._grid_colour = "#9a9a9a" if light else "#5a5a5a"
        return "#0072b2" if light else "#ffcc00"

    def set_background(self, colour: str) -> None:
        """Set the canvas colour, and re-colour the decorations to suit it.

        Parameters
        ----------
        colour
            Any napari colour: a name such as ``"white"``, or ``"#rrggbb"``.
            The axis text, tick lines and gate outlines flip between light and
            dark so they stay legible against it.

        Raises
        ------
        ValueError
            If napari does not recognise the colour.
        """
        gate_edge = self._derive_colours(colour)
        try:
            self.viewer.canvas.background_color_override = self.background
        except AttributeError:  # pragma: no cover - older napari
            canvas = getattr(getattr(self.viewer.window, "_qt_viewer", None), "canvas", None)
            if canvas is not None:
                canvas.bgcolor = self.background

        # Both: `edge_color` recolours the shapes that exist, `current_*` the
        # ones drawn next -- and the grid rebuilds its lines on every redraw.
        self.grid.edge_color = self._grid_colour
        self.grid.current_edge_color = self._grid_colour
        self.gates.edge_color = gate_edge
        self.gates.face_color = gate_edge + "22"
        self.gates.current_edge_color = gate_edge
        self.gates.current_face_color = gate_edge + "22"
        if self.axes is not None:
            self._draw_axes()

        # ------------------------------------------------------------------- data

    def _gate_choices(self) -> list[str]:
        bools = [
            c
            for c in self.adata.obs.columns
            if self.adata.obs[c].dtype == bool or self.adata.obs[c].dtype == "boolean"
        ]
        return ["<none>"] + bools

    # ------------------------------------------------------------------ axes
    def _limits(self, scale: Scale, values: np.ndarray) -> tuple[float, float]:
        q = (0.001, 0.999) if self.w_robust.value else (0.0, 1.0)
        return scale.limits(values, quantiles=q)

    # ------------------------------------------------------------------- data
    def _matrix(self, layer: str):
        return self.adata.X if layer == "X" else self.adata.layers[layer]

    def selection_mask(self, panel: Panel | None = None, *, sample: bool = True) -> np.ndarray:
        """Events a panel plots: its sample, narrowed by its parent gate.

        Parameters
        ----------
        panel
            Which panel; the active one by default.
        sample
            Apply the panel's sample. ``False`` keeps every sample, which is
            how the axis limits stay put as you switch between them.

        Returns
        -------
        ndarray
            Boolean mask over all events.
        """
        panel = panel or self.panel
        mask = np.ones(self.adata.n_obs, dtype=bool)
        if sample and panel.sample != "<all>" and "sample" in self.adata.obs:
            mask &= self.adata.obs["sample"].astype(str).to_numpy() == panel.sample
        if panel.parent != "<none>" and panel.parent in self.adata.obs:
            mask &= self.adata.obs[panel.parent].to_numpy(dtype=bool)
        return mask

    def _column(self, channel: str, mask: np.ndarray, layer: str | None = None) -> np.ndarray:
        """The stored values for ``channel``. Never transformed here."""
        layer = self.panel.layer if layer is None else layer
        j = channel_index(self.adata, channel)
        col = np.asarray(self._matrix(layer)[:, j], dtype=np.float64).ravel()
        return col[mask]

    def _axis_scale(self, channel: str, layer: str | None = None) -> Scale:
        """How to label the axis for ``channel``. Tick placement only."""
        layer = self.panel.layer if layer is None else layer
        if self.w_ticks.value == "linear" or layer == "X" or not channel:
            return LinearScale()
        info = self.adata.uns.get("cytopy", {})
        j = channel_index(self.adata, channel)
        if layer == info.get("asinh_layer") and "cofactor" in self.adata.var:
            cof = float(self.adata.var["cofactor"].iloc[j])
            if np.isfinite(cof):
                return PretransformedScale(AsinhScale(cofactor=cof))
        if layer == info.get("logicle_layer"):
            params = info.get("logicle_params", {}).get(str(self.adata.var_names[j]))
            if params:
                return PretransformedScale(LogicleScale(**params))
        return LinearScale()

    def _describe_transform(self) -> str:
        layer = self.panel.layer
        info = self.adata.uns.get("cytopy", {})
        if self.w_ticks.value == "linear":
            return "ticks: stored values"
        if layer != "X" and layer == info.get("asinh_layer"):
            return "ticks: raw units (arcsinh)"
        if layer != "X" and layer == info.get("logicle_layer"):
            return "ticks: raw units (logicle)"
        return "ticks: stored values (layer records no transform)"

    @property
    def histogram(self) -> bool:
        """Whether the active panel draws distributions rather than a density."""
        return self.panel.histogram

    def histogram_groups(self, panel: Panel | None = None) -> list[tuple[str, np.ndarray]]:
        """``(name, mask)`` per curve: one sample each, within the parent gate.

        Parameters
        ----------
        panel
            Which panel; the active one by default.

        Returns
        -------
        list of tuple
            Empty when the panel's selection holds no events.
        """
        panel = panel or self.panel
        mask = self.selection_mask(panel)
        if not mask.any():
            return []
        if "sample" not in self.adata.obs:
            return [("all", mask)]
        labels = self.adata.obs["sample"].astype(str).to_numpy()
        return [(str(n), mask & (labels == n)) for n in pd.unique(labels[mask])]

    def _draw_curves(self, panel: Panel, mask: np.ndarray) -> float:
        """One smoothed distribution of the panel's x channel per sample."""
        ax = panel.axes
        groups = self.histogram_groups(panel)
        curves, colours, legend = [], [], []
        for index, (name, group) in enumerate(groups):
            values = self._column(panel.x, group, panel.layer)
            curves.append(
                density_curve(
                    values,
                    ax.x_lo,
                    ax.x_hi,
                    self.bins,
                    smooth=max(float(self.w_smooth.value), 0.5) * 2,
                )
            )
            colours.append(CURVE_COLOURS[index % len(CURVE_COLOURS)])
            legend.append(f"{name} ({int(group.sum()):,})")

        if not curves:
            panel.curves.data = []
            return 0.0

        stacked = np.vstack(curves)
        if self.w_peak.value:
            peaks = stacked.max(axis=1, keepdims=True)
            stacked = stacked / np.where(peaks > 0, peaks, 1.0)
            top = 1.0
        else:
            top = float(stacked.max()) or 1.0
        scale = (self.bins - 1) / (top * 1.05)
        columns = np.arange(self.bins, dtype=float) - 0.5
        paths = [np.column_stack([(self.bins - 1) - row * scale, columns]) for row in stacked]

        step = 0.045 * self.bins
        for index in range(len(paths)):
            y = 0.04 * self.bins + index * step
            paths.append(np.array([[y, 0.70 * self.bins], [y, 0.76 * self.bins]]))
            colours.append(colours[index])

        panel.curves.data = []
        panel.curves.add_paths(paths, edge_color=colours, edge_width=2)
        panel.legend = [
            (panel.row + 0.04 * self.bins + i * step, panel.col + 0.78 * self.bins, text)
            for i, text in enumerate(legend)
        ]
        return top

    def _status(self) -> str:
        panel = self.panel
        n = int(self.selection_mask(panel).sum())
        which = f"panel {panel.index}" if len(self.panels) > 1 else ""
        if panel.histogram:
            return f"{which}   {n:,} events   ({panel.x})   {len(self.histogram_groups())} curve(s)"
        peak = ""
        if self.w_colorbar.value:
            smoothed = " smoothed" if self.w_smooth.value else ""
            peak = f"   peak {self.peak_density():,.0f}{smoothed} events/bin"
        return f"{which}   {n:,} events   ({panel.x} × {panel.y}){peak}"

    # -------------------------------------------------------------- rendering
    def refresh(self, *_) -> None:
        """Redraw every panel, and the decorations that go round them."""
        if self._updating:
            return
        self.bins = int(self.w_bins.value)
        self._sync_enabled()
        self.w_transform.value = self._describe_transform()

        drawn = self._shapes_in_data()
        for panel in self.panels:
            self._draw_panel(panel)
        self._restore_shapes(drawn)

        # Panels sitting on the same spot are overlaid, and there the denser
        # one wins each pixel so that neither buries the other.
        for group in self._stacks():
            self._composite(group)

        top = max((p.peak for p in self.panels), default=0.0)
        for panel in self.panels:
            if not panel.histogram:
                panel.image.contrast_limits = (0.0, top or 1.0)

        self._draw_axes()
        self._draw_colorbar(top)
        if getattr(self, "bead_gates", None):
            self._draw_bead_gate()
        self.w_status.value = self._status()
        if self._layout != self._layout_key():
            self._layout = self._layout_key()
            self.viewer.reset_view()

    def _layout_key(self):
        return (len(self.panels), self.bins, tuple((p.row, p.col) for p in self.panels))

    def _stacks(self) -> list[list[Panel]]:
        """Panels grouped by position; a group of more than one is an overlay."""
        groups: dict[tuple[float, float], list[Panel]] = {}
        for panel in self.panels:
            groups.setdefault((round(panel.row, 3), round(panel.col, 3)), []).append(panel)
        return [g for g in groups.values() if len(g) > 1]

    def _composite(self, group: list[Panel]) -> None:
        """Let the densest panel win each pixel where a stack overlaps."""
        images = [p for p in group if not p.histogram and p.image.visible]
        if len(images) < 2:
            return
        stacked = np.stack([np.asarray(p.image.data, dtype=np.float32) for p in images])
        winner = stacked.argmax(axis=0)
        for i, panel in enumerate(images):
            panel.image.data = np.where(winner == i, stacked[i], 0.0).astype(np.float32)

    def _draw_panel(self, panel: Panel) -> None:
        """Compute and draw one panel's density or curves."""
        mask = self.selection_mask(panel)
        panel.x_scale = self._axis_scale(panel.x, panel.layer)
        panel.y_scale = self._axis_scale(panel.y, panel.layer)
        panel.legend = []
        panel.peak = 0.0

        if not mask.any():
            panel.image.data = np.zeros((self.bins, self.bins), dtype=np.float32)
            panel.curves.data = []
            panel.axes = panel.axes or Axes2D(0.0, 1.0, 0.0, 1.0, bins=self.bins)
            return

        xv = self._column(panel.x, mask, panel.layer)
        scope = mask
        if self.w_lock.value and panel.sample != "<all>":
            scope = self.selection_mask(panel, sample=False)
        x_lo, x_hi = self._limits(panel.x_scale, self._column(panel.x, scope, panel.layer))
        if panel.histogram:
            y_lo, y_hi = 0.0, 1.0
        else:
            y_lo, y_hi = self._limits(panel.y_scale, self._column(panel.y, scope, panel.layer))
        panel.axes = Axes2D(x_lo, x_hi, y_lo, y_hi, bins=self.bins)

        if panel.histogram:
            panel.image.visible = False
            panel.curves.visible = True
            panel.peak = self._draw_curves(panel, mask)
            panel.curves.translate = (panel.row, panel.col)
            return

        panel.curves.visible = False
        panel.image.visible = True
        image = density_image(
            xv,
            self._column(panel.y, mask, panel.layer),
            panel.axes,
            smooth=float(self.w_smooth.value),
            log=bool(self.w_log.value),
        )
        panel.image.data = image
        panel.image.colormap = faded_colormap(self.w_cmap.value)
        panel.image.translate = (panel.row, panel.col)
        panel.peak = float(image.max())

    def _draw_colorbar(self, top: float) -> None:
        """A vertical ramp of the colormap, to the right of everything."""
        if not self.w_colorbar.value or self.panel.histogram:
            self.colorbar.visible = False
            return
        bins = self.bins
        width = max(round(bins * _BAR_WIDTH), 3)
        ramp = np.linspace(top or 1.0, 0.0, bins, dtype=np.float32)[:, None]
        self.colorbar.data = np.repeat(ramp, width, axis=1)
        self.colorbar.contrast_limits = (0.0, top or 1.0)
        self.colorbar.colormap = faded_colormap(self.w_cmap.value)
        self.colorbar.translate = (self._bar_row(), self._bar_col())
        self.colorbar.visible = True

    def _bar_row(self) -> float:
        return min(p.row for p in self.panels)

    def _bar_col(self) -> float:
        return max(p.col for p in self.panels) + self.bins * (1.0 + _PANEL_GAP / 2)

    def _colorbar_labels(self):
        """Tick positions and labels for the colour bar, as a share of the peak.

        Events per bin would be the obvious thing to write here and it is the
        wrong one: a bin is a cell of the display grid, not a unit of the data.
        The same events in the same channels peak at 470 per bin at 64 bins and
        at 5 per bin at 1024, and changing channel moves it again, so the
        numbers lurch about for reasons that have nothing to do with the
        sample. A share of the peak holds still, and stays comparable between
        panels because they share one scale. The peak itself is in the status
        line, where it does not move around.
        """
        if not self.w_colorbar.value or self.panel.histogram:
            return [], []
        bins = self.bins
        width = max(round(bins * _BAR_WIDTH), 3)
        left, top = self._bar_col(), self._bar_row()
        coords, text = [], []
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            coords.append([top + (1.0 - fraction) * (bins - 1), left + width + 0.04 * bins])
            text.append(f"{round(fraction * 100)}%")
        coords.append([top - 0.06 * bins, left + width / 2])
        text.append("of peak")
        return coords, text

    def peak_density(self) -> float:
        """Events in the densest bin of the plot, smoothing included.

        The absolute number behind the colour bar's percentages. It depends on
        the bin count as much as on the data, which is why it is reported once
        rather than used to label the bar.

        Returns
        -------
        float
            ``0.0`` before anything has been drawn.
        """
        top = float(self.density.contrast_limits[1])
        return float(np.expm1(top)) if self.w_log.value else top

    def _sync_enabled(self) -> None:
        """Grey out the settings the active panel has no use for."""
        histogram = self.panel.histogram
        for widget in (self.w_y, self.w_swap, self.w_colorbar, self.w_log, self.w_cmap):
            widget.enabled = not histogram
        self.w_peak.enabled = histogram
        self.w_remove.enabled = len(self.panels) > 1

    def _draw_axes(self) -> None:
        """Draw a labelled frame around every panel, and the colour bar's ticks."""
        b = self.bins
        tick = 0.02 * b
        lines: list[np.ndarray] = []
        coords: list[list[float]] = []
        text: list[str] = []
        many = len(self.panels) > 1

        for panel in self.panels:
            ax = panel.axes
            if ax is None:
                continue
            dr, dc = panel.row, panel.col
            lines += [
                np.array([[dr - 0.5, dc - 0.5], [dr - 0.5, dc + b - 0.5]]),
                np.array([[dr + b - 0.5, dc - 0.5], [dr + b - 0.5, dc + b - 0.5]]),
                np.array([[dr - 0.5, dc - 0.5], [dr + b - 0.5, dc - 0.5]]),
                np.array([[dr - 0.5, dc + b - 0.5], [dr + b - 0.5, dc + b - 0.5]]),
            ]
            xt = panel.x_scale.ticks(ax.x_lo, ax.x_hi)
            for pos, lab in zip(xt.major, xt.labels):
                col = dc + ax.to_pixels(np.array([pos]), np.array([ax.y_lo]))[0, 1]
                lines.append(np.array([[dr + b - 0.5, col], [dr + b - 0.5 + tick, col]]))
                coords.append([dr + b + 3.0 * tick, col])
                text.append(lab)
            for pos in xt.minor:
                col = dc + ax.to_pixels(np.array([pos]), np.array([ax.y_lo]))[0, 1]
                lines.append(np.array([[dr + b - 0.5, col], [dr + b - 0.5 + tick / 2, col]]))
            coords.append([dr + b + 8.0 * tick, dc + b / 2.0])
            text.append(str(panel.x))

            if panel.histogram:
                # The y axis is a density, not a channel.
                for fraction in (0.0, 0.5, 1.0):
                    row = dr + (b - 1) - fraction * (b - 1) / 1.05
                    lines.append(np.array([[row, dc - 0.5], [row, dc - 0.5 - tick]]))
                    coords.append([row, dc - 4.0 * tick])
                    text.append(f"{fraction * panel.peak:.3g}")
                coords.append([dr + b / 2.0, dc - 12.0 * tick])
                text.append("peak" if self.w_peak.value else "density")
                for row, col, label in panel.legend:
                    coords.append([row, col])
                    text.append(label)
            else:
                yt = panel.y_scale.ticks(ax.y_lo, ax.y_hi)
                for pos, lab in zip(yt.major, yt.labels):
                    row = dr + ax.to_pixels(np.array([ax.x_lo]), np.array([pos]))[0, 0]
                    lines.append(np.array([[row, dc - 0.5], [row, dc - 0.5 - tick]]))
                    coords.append([row, dc - 4.0 * tick])
                    text.append(lab)
                for pos in yt.minor:
                    row = dr + ax.to_pixels(np.array([ax.x_lo]), np.array([pos]))[0, 0]
                    lines.append(np.array([[row, dc - 0.5], [row, dc - 0.5 - tick / 2]]))
                coords.append([dr + b / 2.0, dc - 12.0 * tick])
                text.append(str(panel.y))

            if many:
                marker = "> " if panel is self.panel else ""
                coords.append([dr - 3.0 * tick, dc + b / 2.0])
                text.append(f"{marker}{panel.index}: {panel.title}")

        bar_coords, bar_text = self._colorbar_labels()
        coords += bar_coords
        text += bar_text

        self.grid.data = []
        if lines:
            self.grid.add_lines(lines)
        # Assign the strings directly rather than through a feature column: an
        # encoding is briefly out of step with the new point count and napari
        # silently falls back to blank labels.
        self.labels.data = np.asarray(coords, dtype=float) if coords else np.empty((0, 2))
        self.labels.text = {**_TEXT_STYLE, "string": text, "color": self._foreground}

    # --------------------------------------------------------------- callbacks
    def _on_change(self, *_):
        if self._updating:
            return
        self.refresh()

    def _swap(self, *_):
        self._updating = True
        try:
            self.w_x.value, self.w_y.value = self.w_y.value, self.w_x.value
        finally:
            self._updating = False
        self.refresh()

    # ------------------------------------------------------------------ gating
    def current_gate_mask(self) -> np.ndarray:
        """Mask over *all* events of those inside the shapes drawn on the canvas.

        On a histogram there is no second channel to be inside of, so a shape
        gates the interval it spans on the x axis -- drag a box over a peak and
        you get the events under it, which is how a threshold is set.

        Returns
        -------
        ndarray
            Boolean mask over all events, all ``False`` when nothing is drawn.
        """
        out = np.zeros(self.adata.n_obs, dtype=bool)
        if self.axes is None or not len(self.gates.data):
            return out
        sel = self.selection_mask()
        x = self._column(self.w_x.value, sel)

        if self.histogram:
            inside = np.zeros(x.shape[0], dtype=bool)
            for shape in self.gates.data:
                columns = np.asarray(shape, dtype=float)[:, 1]
                lo, hi = self.axes.to_display(
                    np.column_stack([np.zeros(2), [columns.min(), columns.max()]])
                )[:, 0]
                inside |= (x >= lo) & (x <= hi)
        else:
            pts = self.axes.to_pixels(x, self._column(self.w_y.value, sel))
            inside = shapes_mask(pts, self.gates.data, self.gates.shape_type)

        out[np.flatnonzero(sel)] = inside
        return out

    def _shapes_in_data(self):
        """The drawn shapes in data coordinates, before the axes move under them."""
        if self.axes is None or not len(self.gates.data):
            return None
        return (
            [self.axes.to_display(np.asarray(v, dtype=float)) for v in self.gates.data],
            [str(k) for k in self.gates.shape_type],
        )

    def _restore_shapes(self, drawn) -> None:
        """Redraw shapes so they keep covering the same data after a rescale.

        A shape lives on the canvas in pixels, but it means a region of the
        data. Changing channel, sample, parent gate or bin count moves the
        axes underneath it, and without this the shape would silently come to
        mean something else -- the same pixels over different values.
        """
        if drawn is None or self.axes is None:
            return
        shapes, kinds = drawn
        pixels = [self.axes.to_pixels(v[:, 0], v[:, 1]) for v in shapes]
        self.gates.data = []
        for verts, kind in zip(pixels, kinds):
            self.gates.add(verts, shape_type=kind)

    def off_axis_count(self) -> int:
        """Events in the current selection that fall outside the plotted axes.

        Robust limits clip the axes to the 0.1-99.9th percentile, so a few
        events are drawn nowhere. They cannot be gated either, which is worth
        knowing when a gate around everything visible comes back smaller than
        the population it was drawn on.

        Returns
        -------
        int
            Number of selected events outside the axis range, ``0`` when the
            plot has not been drawn yet.
        """
        if self.axes is None:
            return 0
        sel = self.selection_mask()
        x = self._column(self.w_x.value, sel)
        y = self._column(self.w_y.value, sel)
        inside = (
            (x >= self.axes.x_lo)
            & (x <= self.axes.x_hi)
            & (y >= self.axes.y_lo)
            & (y <= self.axes.y_hi)
        )
        return int((~inside).sum())

    def apply_gate(self, name: str, *, parent: str | None = None) -> np.ndarray:
        """Turn the shapes on the canvas into a named gate.

        What the *apply gate from shapes* button does, callable directly.

        Parameters
        ----------
        name
            ``obs`` column to write. An existing gate of the same name is
            overwritten.
        parent
            Gate to nest inside, so the new one only ever contains events the
            parent already held. ``None`` gates the current selection.

        Returns
        -------
        ndarray
            Boolean mask over all events.

        Raises
        ------
        ValueError
            If nothing has been drawn on the canvas.
        """
        if not len(self.gates.data):
            raise ValueError("draw a shape on the canvas first")
        self._updating = True
        try:
            self.w_gate_name.value = name
            self.w_parent.value = parent or "<none>"
        finally:
            self._updating = False
        self.refresh()
        self._apply_gate()
        return self.adata.obs[name].to_numpy(dtype=bool)

    def _apply_gate(self, *_):
        name = (self.w_gate_name.value or "").strip()
        if not name:
            self.w_status.value = "give the gate a name first"
            return
        if not len(self.gates.data):
            self.w_status.value = "draw a shape on the canvas first"
            return
        parent = None if self.w_parent.value == "<none>" else self.w_parent.value
        mask = add_gate(
            self.adata,
            name,
            self.current_gate_mask(),
            parent=parent,
            meta={
                "x": str(self.w_x.value),
                "y": str(self.w_y.value),
                "layer": str(self.w_layer.value),
                # In data coordinates, so the gate can be redrawn in a report
                # long after the canvas it was drawn on is gone.
                "vertices": [
                    self.axes.to_display(np.asarray(shape, dtype=float)).tolist()
                    for shape in self.gates.data
                ],
                "shape_types": [str(k) for k in self.gates.shape_type],
            },
        )
        n = int(mask.sum())
        denom = int(self.selection_mask().sum()) or 1
        note = ""
        off = self.off_axis_count()
        if off:
            note = f"; {off:,} outside the axes, not gated"
        # Clear, so the next gate is only what is drawn for it. Without this
        # every gate is the union of everything ever drawn on the canvas.
        self.gates.data = []
        self.w_status.value = f"{name}: {n:,} events ({100 * n / denom:.2f}% of parent){note}"
        self._updating = True
        try:
            self.w_parent.choices = self._gate_choices()
        finally:
            self._updating = False


def as_one_anndata(data, *, names: Sequence[str] | None = None) -> ad.AnnData:
    """Gather whatever was passed into a single AnnData with a ``sample`` column.

    The viewer plots one matrix; several files become several values of
    ``obs['sample']``, which the **sample** selector then switches between.

    Parameters
    ----------
    data
        An ``AnnData``; a path to an FCS file, an ``.h5ad`` or a directory of
        FCS files; a sequence of any of those; or a mapping from the name you
        want in the selector to any of those.
    names
        Names for the samples, overriding whatever they carry. Must be one per
        object.

    Returns
    -------
    AnnData
        A single object. One input is returned untouched; several are
        concatenated on the channels they share, with duplicate sample names
        made unique so two files never merge into one entry.

    Raises
    ------
    ValueError
        If nothing was passed, or ``names`` is the wrong length.
    TypeError
        If an entry is neither an AnnData nor a path.
    """
    from .io import concat_samples

    if isinstance(data, ad.AnnData):
        parts, labels = [data], list(names) if names else [None]
    elif isinstance(data, Mapping):
        parts, labels = list(data.values()), [str(k) for k in data]
        if names:
            labels = list(names)
    elif isinstance(data, str | os.PathLike):
        parts, labels = [data], list(names) if names else [None]
    else:
        parts = list(data)
        labels = list(names) if names else [None] * len(parts)

    if not parts:
        raise ValueError("nothing to view")
    if len(labels) != len(parts):
        raise ValueError(f"{len(parts)} objects but {len(labels)} names")

    loaded = [_load(part) for part in parts]
    if len(loaded) == 1 and labels[0] is None:
        return loaded[0]

    seen: dict[str, int] = {}
    for adata, label in zip(loaded, labels):
        name = label if label is not None else _sample_name(adata)
        # Two files called the same thing must not collapse into one entry.
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        adata.obs["sample"] = pd.Categorical([name] * adata.n_obs)
    return concat_samples(loaded) if len(loaded) > 1 else loaded[0]


def _load(part):
    """An AnnData, read from disk if a path was given."""
    if isinstance(part, ad.AnnData):
        return part
    if not isinstance(part, str | os.PathLike):
        raise TypeError(f"expected an AnnData or a path, got {type(part).__name__}")
    from .io import read_fcs, read_fcs_dir

    path = Path(part)
    if path.is_dir():
        return read_fcs_dir(path)
    if path.suffix == ".h5ad":
        return ad.read_h5ad(path)
    return read_fcs(path)


def _sample_name(adata: ad.AnnData) -> str:
    """Whatever this object already calls itself."""
    if "sample" in adata.obs and adata.n_obs:
        return str(adata.obs["sample"].astype(str).iloc[0])
    return "sample"


def _is_light(colour) -> bool:
    """Whether text should be drawn dark on this background."""
    from napari.utils.colormaps.standardize_color import transform_color

    red, green, blue = transform_color(colour)[0][:3]
    return float(0.2126 * red + 0.7152 * green + 0.0722 * blue) > 0.5


def faded_colormap(name: str, fade: float = _FADE):
    """A colormap whose bottom end is transparent rather than its darkest colour.

    The density image covers the whole plot, so without this the background is
    whatever the colormap maps zero counts to -- for ``turbo`` that is a dark
    blue-purple, which is what makes an untouched napari canvas look blue. With
    the low end faded, empty bins show the canvas through instead.

    Parameters
    ----------
    name
        Any colormap napari knows, e.g. one of :data:`COLORMAPS`.
    fade
        Fraction of the colormap that ramps from transparent to opaque. Small:
        a bin holding a single event should still be clearly visible.

    Returns
    -------
    napari.utils.colormaps.Colormap
        The same colours, with alpha ramped in at the bottom.
    """
    import numpy as np
    from napari.utils.colormaps import Colormap, ensure_colormap

    colours = np.array(ensure_colormap(name).colors, dtype=np.float32, copy=True)
    steps = max(round(len(colours) * float(fade)), 2)
    colours[:steps, 3] = np.linspace(0.0, 1.0, steps)
    return Colormap(colors=colours, name=f"{name}_faded", display_name=name)


def _hide_overlays(viewer) -> None:
    """Turn off napari's own axis cross and scale bar, across napari versions."""
    for path in (("scene", "overlays", "axes"), ("canvas", "overlays", "scale_bar")):
        target = viewer
        for attr in path:
            target = getattr(target, attr, None)
            if target is None:
                break
        if target is not None:
            target.visible = False
            continue
        legacy = getattr(viewer, path[-1], None)  # napari < 0.9
        if legacy is not None:
            legacy.visible = False


def view(adata, *, block: bool | None = None, **kwargs) -> CytoViewer:
    """Open the cytopy napari window on one or several samples.

    Plots the chosen layer as stored — run any transform you want first.

    Several objects are concatenated on the channels they share and become
    entries in the **sample** selector, so one window browses the lot::

        cytopy.view([run1, run2, run3])
        cytopy.view({"healthy": run1, "treated": run2})
        cytopy.view("data/")                       # every FCS in a directory

    Parameters
    ----------
    adata
        What to plot. One events x channels AnnData, e.g. from
        :func:`cytopy.read_fcs`; a path to an FCS file, an ``.h5ad`` or a
        directory; or several of either as a list or a ``{name: object}``
        mapping.
    block
        Start the Qt event loop and return only once the window closes. The
        default detects the context: ``True`` from a script, ``False`` under
        IPython/Jupyter where a loop is already running. Gates drawn in the
        window are on ``adata.obs`` by the time a blocking call returns.
    **kwargs
        Passed to :class:`CytoViewer` — ``names``, ``layer``, ``x``, ``y``,
        ``ticks``, ``bins``, ``colormap``, ``smooth``, ``robust``,
        ``background``, ``viewer``, ``title``.

    Returns
    -------
    CytoViewer
        The viewer, so gates and the current selection stay reachable
        afterwards.
    """
    import napari

    cv = CytoViewer(adata, **kwargs)
    if block is None:
        try:
            get_ipython  # type: ignore[name-defined]  # noqa: B018
            block = False
        except NameError:
            block = True
    if block:
        napari.run()
    return cv
