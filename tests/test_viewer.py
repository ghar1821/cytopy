"""Viewer tests. Run headless with QT_QPA_PLATFORM=offscreen."""

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")


@pytest.fixture
def cv(demo, make_napari_viewer):
    import cytopy

    cytopy.asinh_transform(demo, 150.0)
    from cytopy.viewer import CytoViewer

    return CytoViewer(
        demo,
        layer="asinh",
        x="CD3 (FITC-A)",
        y="CD19 (PE-A)",
        bins=128,
        viewer=make_napari_viewer(),
    )


def test_layers_exist(cv):
    assert set(cv.viewer.layers) >= set()
    names = [layer.name for layer in cv.viewer.layers]
    assert names == [
        "colour bar",
        "axes",
        "axis labels",
        "gates",
        "bead gate",
        "panel 1",
        "panel 1 curves",
    ]


def test_density_image_is_populated(cv):
    img = cv.density.data
    assert img.shape == (128, 128)
    assert img.max() > 0
    assert np.isfinite(img).all()


def _labels(cv):
    return list(cv.viewer.layers["axis labels"].text.string.array)


def test_axis_labels_are_raw_units_for_a_transformed_layer(cv):
    text = _labels(cv)
    assert "CD3 (FITC-A)" in text and "CD19 (PE-A)" in text
    assert "0" in text
    assert any(t.startswith("10") for t in text)


def test_switching_channels_redraws(cv):
    before = cv.density.data.copy()
    cv.w_y.value = "CD8 (APC-A)"
    assert not np.array_equal(before, cv.density.data)


def test_linear_ticks_label_the_stored_values(cv):
    from cytopy.scales import LinearScale, PretransformedScale

    assert isinstance(cv.x_scale, PretransformedScale)
    cv.w_ticks.value = "linear"
    cv.w_colorbar.value = False  # its "100%" is not an axis label
    assert isinstance(cv.x_scale, LinearScale)
    # stored values are single-digit arcsinh units, not decades of raw signal
    assert not any(lbl.startswith("10") for lbl in _labels(cv))


def test_the_viewer_never_transforms_the_data(cv):
    """What is plotted must be exactly what is stored in the layer."""
    import cytopy

    mask = cv.selection_mask()
    j = cytopy.channel_index(cv.adata, "CD3 (FITC-A)")
    assert np.allclose(cv._column("CD3 (FITC-A)", mask), cv.adata.layers["asinh"][mask, j])
    # ... and switching the tick mode must not touch the values either
    cv.w_ticks.value = "linear"
    assert np.allclose(cv._column("CD3 (FITC-A)", mask), cv.adata.layers["asinh"][mask, j])


def test_raw_X_gets_linear_ticks(cv):
    from cytopy.scales import LinearScale

    cv.w_layer.value = "X"
    assert isinstance(cv.x_scale, LinearScale)
    assert "records no transform" in cv.w_transform.value


def test_swap_axes(cv):
    x, y = cv.w_x.value, cv.w_y.value
    cv.w_swap.clicked()
    assert (cv.w_x.value, cv.w_y.value) == (y, x)


def test_axes_roundtrip_pixels_and_display(cv):
    disp = np.array([[0.2, 0.3], [0.8, 0.55]])
    rc = cv.axes.to_pixels(disp[:, 0], disp[:, 1])
    back = cv.axes.to_display(rc)
    assert np.allclose(back, disp, atol=1e-6)


def test_gate_from_a_rectangle(cv):
    """A rectangle over the top-left quadrant must select the CD3-/CD19+ cells."""
    b = cv.bins
    cv.viewer.layers["gates"].add_rectangles(
        np.array([[0, 0], [0, b / 2], [b / 2, b / 2], [b / 2, 0]])
    )
    cv.w_gate_name.value = "B cells"
    cv.w_gate_apply.clicked()

    mask = cv.adata.obs["B cells"].to_numpy(dtype=bool)
    assert 0 < mask.sum() < cv.adata.n_obs
    # top-left of the canvas is low x (CD3) and high y (CD19)
    import cytopy

    cd3 = cv.adata.layers["asinh"][:, cytopy.channel_index(cv.adata, "CD3")]
    cd19 = cv.adata.layers["asinh"][:, cytopy.channel_index(cv.adata, "CD19")]
    assert cd19[mask].mean() > cd19[~mask].mean()
    assert cd3[mask].mean() < cd3[~mask].mean()
    assert cv.adata.uns["cytopy"]["gates"]["B cells"]["x"] == "CD3 (FITC-A)"


def test_gate_hierarchy_intersects_with_parent(cv):
    b = cv.bins
    cv.w_robust.value = False  # so a full-canvas rectangle really does catch everything
    gates = cv.viewer.layers["gates"]
    gates.add_rectangles(np.array([[0, 0], [0, b], [b, b], [b, 0]]))
    cv.w_gate_name.value = "all"
    cv.w_gate_apply.clicked()
    assert cv.adata.obs["all"].sum() == cv.adata.n_obs

    gates.data = []
    gates.add_rectangles(np.array([[0, 0], [0, b / 2], [b / 2, b / 2], [b / 2, 0]]))
    cv.w_parent.value = "all"
    cv.w_gate_name.value = "child"
    cv.w_gate_apply.clicked()
    child = cv.adata.obs["child"].to_numpy(dtype=bool)
    assert child.sum() > 0
    assert (child & ~cv.adata.obs["all"].to_numpy(dtype=bool)).sum() == 0
    assert cv.adata.uns["cytopy"]["gates"]["child"]["parent"] == "all"


def test_parent_gate_subsets_the_plotted_events(cv):
    b = cv.bins
    cv.viewer.layers["gates"].add_rectangles(
        np.array([[0, 0], [0, b / 2], [b / 2, b / 2], [b / 2, 0]])
    )
    cv.w_gate_name.value = "g"
    cv.w_gate_apply.clicked()
    total = cv.selection_mask().sum()
    cv.w_parent.value = "g"
    assert cv.selection_mask().sum() < total
    assert cv.selection_mask().sum() == cv.adata.obs["g"].sum()


def test_gate_without_shapes_is_a_no_op(cv):
    cv.w_gate_name.value = "empty"
    cv.w_gate_apply.clicked()
    assert "empty" not in cv.adata.obs
    assert "draw a shape" in cv.w_status.value


def test_robust_limits_trim_the_axis_range(cv):
    cv.w_robust.value = True
    tight = cv.axes.x_hi - cv.axes.x_lo
    cv.w_robust.value = False
    assert cv.axes.x_hi - cv.axes.x_lo > tight


def test_channels_can_be_named_by_marker_or_detector(demo, make_napari_viewer):
    import cytopy
    from cytopy.viewer import CytoViewer

    cytopy.asinh_transform(demo, 150.0)
    cv = CytoViewer(demo, layer="asinh", x="CD3", y="PE-A", bins=64, viewer=make_napari_viewer())
    assert (cv.w_x.value, cv.w_y.value) == ("CD3 (FITC-A)", "CD19 (PE-A)")


# --------------------------------------------------------------------------
# the bead panel
# --------------------------------------------------------------------------
@pytest.fixture
def bead_cv(cytof, make_napari_viewer):
    from cytopy.viewer import CytoViewer

    return CytoViewer(cytof, bins=128, viewer=make_napari_viewer())


def test_no_bead_panel_without_bead_channels(cv):
    """A flow panel has no bead channels, so the panel is not built at all."""
    assert cv.bead_gates is None
    assert not hasattr(cv, "w_bead_apply")


def test_bead_panel_starts_on_premessas_gate(bead_cv):
    assert bead_cv.dna == "DNA2 (Ir193Di)"
    assert list(bead_cv.bead_gates) == ["Ce140Di", "Eu151Di", "Eu153Di", "Ho165Di", "Lu175Di"]
    assert (bead_cv.w_bead_xlo.value, bead_cv.w_bead_xhi.value) == (2.0, 5.0)
    assert (bead_cv.w_bead_ylo.value, bead_cv.w_bead_yhi.value) == (-1.0, 2.0)


def test_the_panel_does_nothing_until_asked(bead_cv):
    """It builds an arcsinh layer, so it waits to be told to."""
    assert list(bead_cv.adata.layers) == [None, "raw"]
    assert bead_cv.w_layer.value == "X"
    assert not bead_cv.viewer.layers["bead gate"].visible


def test_picking_a_bead_channel_plots_it_against_dna(bead_cv):
    from cytopy.viewer import BEAD_LAYER

    bead_cv.w_bead_channel.value = "Eu151Di"
    assert bead_cv.w_layer.value == BEAD_LAYER
    assert (bead_cv.w_x.value, bead_cv.w_y.value) == ("Eu151Di", "DNA2 (Ir193Di)")
    assert bead_cv.viewer.layers["bead gate"].visible
    assert len(bead_cv.viewer.layers["bead gate"].data) == 1


def test_editing_the_gate_updates_the_count_and_the_outline(bead_cv):
    bead_cv._bead_channel_changed()  # the "plot bead channel vs DNA" button
    before = int(bead_cv.bead_mask().sum())
    outline = np.asarray(bead_cv.viewer.layers["bead gate"].data[0]).copy()

    bead_cv.w_bead_xlo.value = 4.0
    assert bead_cv.bead_gates["Ce140Di"]["x"] == (4.0, 5.0)
    assert int(bead_cv.bead_mask().sum()) < before
    assert "beads" in bead_cv.w_bead_status.value
    assert not np.allclose(np.asarray(bead_cv.viewer.layers["bead gate"].data[0]), outline)


def test_gate_from_a_drawn_rectangle(bead_cv):
    """The premessa gesture: drag a box round the beads, adopt it."""
    bead_cv._bead_channel_changed()
    corners = bead_cv.axes.to_pixels(
        np.array([2.5, 4.5, 4.5, 2.5]), np.array([-0.5, -0.5, 1.0, 1.0])
    )
    bead_cv.gates.add_rectangles([corners])
    bead_cv._bead_gate_from_shape()

    x_lo, x_hi = bead_cv.bead_gates["Ce140Di"]["x"]
    y_lo, y_hi = bead_cv.bead_gates["Ce140Di"]["y"]
    assert (x_lo, x_hi) == pytest.approx((2.5, 4.5), abs=0.05)
    assert (y_lo, y_hi) == pytest.approx((-0.5, 1.0), abs=0.05)


def test_apply_and_normalise_from_the_panel(bead_cv):
    bead_cv._apply_bead_gate()
    assert "bead" in bead_cv.adata.obs
    assert bead_cv.adata.obs["bead"].sum() > 100
    assert "bead" in bead_cv.w_parent.choices

    bead_cv._normalise_beads()
    assert "normalised" in bead_cv.adata.layers
    assert bead_cv.w_layer.value == "normalised"
    assert "slope" in bead_cv.w_bead_status.value


def test_a_gate_holding_nothing_is_reported_not_raised(bead_cv):
    bead_cv._bead_channel_changed()
    bead_cv.w_bead_xlo.value = 19.0
    bead_cv._apply_bead_gate()
    assert "no events" in bead_cv.w_bead_status.value
    assert "bead" not in bead_cv.adata.obs


def test_live_count_is_estimated_from_a_subsample(cytof, make_napari_viewer, monkeypatch):
    """Dragging a gate edge must not re-scan every event on every step."""
    import cytopy.viewer as viewer_module
    from cytopy.viewer import CytoViewer

    monkeypatch.setattr(viewer_module, "BEAD_PREVIEW_EVENTS", 5_000)
    cv = CytoViewer(cytof, bins=128, viewer=make_napari_viewer())
    cv._bead_channel_changed()

    cv.w_bead_xlo.value = 2.1
    assert cv._bead_sample.n_obs == 5_000
    assert "est." in cv.w_bead_status.value

    sampled = cv._bead_sample
    cv.w_bead_xlo.value = 2.2
    assert cv._bead_sample is sampled  # drawn once, reused as the gate moves

    exact = int(cv.bead_mask().sum())
    reported = int(cv.w_bead_status.value.split()[0].replace(",", ""))
    assert reported == pytest.approx(exact, rel=0.15)

    # Applying the gate is exact, over every event, and says so.
    cv._apply_bead_gate()
    assert "est." not in cv.w_bead_status.value
    assert int(cv.adata.obs["bead"].sum()) == exact


def test_small_data_is_counted_exactly(cytof, make_napari_viewer):
    from cytopy.viewer import CytoViewer

    cv = CytoViewer(cytof, bins=128, viewer=make_napari_viewer())
    cv._bead_channel_changed()
    cv.w_bead_xlo.value = 2.1
    assert cv._bead_sample is cv.adata  # 40k events: no need to subsample
    assert "est." not in cv.w_bead_status.value


# --------------------------------------------------------------------------
# canvas background
# --------------------------------------------------------------------------
def test_background_is_white_by_default(cv):
    from cytopy.viewer import DEFAULT_BACKGROUND

    assert DEFAULT_BACKGROUND == "white"
    assert cv.background == "white"
    assert np.allclose(cv.viewer.canvas.background_color_override, [1.0, 1.0, 1.0, 1.0])


def test_empty_bins_are_transparent(cv):
    """Otherwise the background is whatever the colormap maps zero counts to.

    turbo's zero is a dark blue-purple, and the density image covers the whole
    plot, so without this the canvas colour would never be visible.
    """
    colormap = cv.density.colormap
    assert colormap.map(np.array([0.0]))[0][3] == 0.0
    assert colormap.map(np.array([1.0]))[0][3] == 1.0
    # ... and the colours themselves are untouched.
    from napari.utils.colormaps import ensure_colormap

    assert np.allclose(
        colormap.map(np.array([0.5]))[0][:3], ensure_colormap("turbo").map(np.array([0.5]))[0][:3]
    )


def test_decorations_flip_with_the_background(demo, make_napari_viewer):
    from cytopy.viewer import CytoViewer

    cytopy = __import__("cytopy")
    cytopy.asinh_transform(demo, 150.0)
    cv = CytoViewer(demo, layer="asinh", bins=64, viewer=make_napari_viewer())

    def text_colour():
        return np.asarray(
            cv.labels.text.color.constant
            if hasattr(cv.labels.text.color, "constant")
            else cv.labels.text.color
        )[:3]

    light_text = text_colour().copy()
    light_grid = np.asarray(cv.grid.edge_color[0])[:3].copy()

    cv.set_background("black")
    assert cv.background == "black"
    assert text_colour().mean() > light_text.mean()  # dark bg -> light text
    assert np.asarray(cv.grid.edge_color[0])[:3].mean() != light_grid.mean()


def test_background_can_be_changed_from_the_widget(cv):
    cv.w_background.value = "black"
    assert cv.background == "black"
    assert np.allclose(cv.viewer.canvas.background_color_override, [0.0, 0.0, 0.0, 1.0])
    assert "white" in list(cv.w_background.choices)


def test_a_custom_background_is_accepted_and_nonsense_is_not(demo, make_napari_viewer):
    from cytopy.viewer import CytoViewer

    __import__("cytopy").asinh_transform(demo, 150.0)
    cv = CytoViewer(demo, layer="asinh", bins=64, background="#f5f5f5", viewer=make_napari_viewer())
    assert cv.background == "#f5f5f5"
    assert "#f5f5f5" in list(cv.w_background.choices)
    with pytest.raises(ValueError):
        cv.set_background("not a colour")
    assert cv.background == "#f5f5f5"  # unchanged after a bad one


def test_changing_the_colormap_keeps_the_fade(cv):
    cv.w_cmap.value = "viridis"
    assert cv.density.colormap.map(np.array([0.0]))[0][3] == 0.0
    assert "viridis" in cv.density.colormap.name


# --------------------------------------------------------------------------
# several samples in one window
# --------------------------------------------------------------------------
def test_as_one_anndata_leaves_a_single_object_alone(demo):
    from cytopy.viewer import as_one_anndata

    assert as_one_anndata(demo) is demo


def test_as_one_anndata_concatenates_a_list(demo, demo_path):
    import cytopy
    from cytopy.viewer import as_one_anndata

    second = cytopy.read_fcs(demo_path, sample_id="run2")
    out = as_one_anndata([demo, second])
    assert out.n_obs == demo.n_obs + second.n_obs
    assert sorted(out.obs["sample"].astype(str).unique()) == ["demo", "run2"]


def test_as_one_anndata_takes_names_from_a_mapping(demo, demo_path):
    import cytopy
    from cytopy.viewer import as_one_anndata

    second = cytopy.read_fcs(demo_path)
    out = as_one_anndata({"healthy": demo, "treated": second})
    assert sorted(out.obs["sample"].astype(str).unique()) == ["healthy", "treated"]

    out = as_one_anndata([demo, second], names=["a", "b"])
    assert sorted(out.obs["sample"].astype(str).unique()) == ["a", "b"]


def test_same_named_files_do_not_merge(demo, demo_path):
    """Two files both called 'demo' must stay two entries in the selector."""
    import cytopy
    from cytopy.viewer import as_one_anndata

    out = as_one_anndata([demo, cytopy.read_fcs(demo_path)])
    assert sorted(out.obs["sample"].astype(str).unique()) == ["demo", "demo.1"]
    assert out.n_obs == 2 * demo.n_obs


def test_as_one_anndata_reads_paths(demo_path, tmp_path):
    import cytopy
    from cytopy.viewer import as_one_anndata

    assert as_one_anndata(demo_path).n_obs == 60_000
    assert as_one_anndata([demo_path, demo_path]).n_obs == 120_000

    h5 = tmp_path / "demo.h5ad"
    cytopy.read_fcs(demo_path).write_h5ad(h5)
    assert as_one_anndata(h5).n_obs == 60_000

    import shutil

    shutil.copy(demo_path, tmp_path / "a.fcs")
    shutil.copy(demo_path, tmp_path / "b.fcs")
    assert as_one_anndata(tmp_path).n_obs == 120_000  # a directory of FCS


def test_as_one_anndata_rejects_nonsense(demo):
    from cytopy.viewer import as_one_anndata

    with pytest.raises(ValueError, match="nothing to view"):
        as_one_anndata([])
    with pytest.raises(ValueError, match="2 objects but 1 names"):
        as_one_anndata([demo, demo], names=["only one"])
    with pytest.raises(TypeError, match="expected an AnnData or a path"):
        as_one_anndata([demo, 42])


def test_viewer_opens_on_several_samples(demo, demo_path, make_napari_viewer):
    import cytopy
    from cytopy.viewer import CytoViewer

    cytopy.asinh_transform(demo, 150.0)
    second = cytopy.read_fcs(demo_path, sample_id="run2")
    cytopy.asinh_transform(second, 150.0)

    cv = CytoViewer(
        [demo, second],
        layer="asinh",
        x="CD3",
        y="CD19",
        bins=64,
        viewer=make_napari_viewer(),
    )
    assert cv.adata.n_obs == 120_000
    assert list(cv.w_sample.choices) == ["<all>", "demo", "run2"]
    assert cv.w_lock.value is True  # locked once there is more than one

    cv.w_sample.value = "run2"
    assert cv.panel.sample == "run2"
    assert int(cv.selection_mask().sum()) == 60_000
    assert int(cv.selection_mask(sample=False).sum()) == 120_000


def test_axes_stay_put_between_samples(demo, demo_path, make_napari_viewer):
    """Rescaling per sample is what makes two samples look alike when they are not."""
    import cytopy
    from cytopy.viewer import CytoViewer

    dim = cytopy.read_fcs(demo_path, sample_id="dim")
    dim.X = dim.X * np.float32(0.25)
    cv = CytoViewer([demo, dim], x="CD3", y="CD19", bins=64, viewer=make_napari_viewer())

    cv.w_sample.value = "demo"
    locked = (cv.axes.x_lo, cv.axes.x_hi)
    cv.w_sample.value = "dim"
    assert (cv.axes.x_lo, cv.axes.x_hi) == locked  # same axes, so they compare

    cv.w_lock.value = False
    assert cv.axes.x_hi < locked[1]  # unlocked, the dim sample fills the plot


def test_a_single_sample_is_not_locked_by_default(cv):
    assert list(cv.w_sample.choices) == ["<all>", "demo"]
    assert cv.w_lock.value is False


# --------------------------------------------------------------------------
# gate hierarchies
# --------------------------------------------------------------------------
def _rect(cv, x_lo, x_hi, y_lo, y_hi):
    """A rectangle given in data coordinates, as canvas pixels."""
    return cv.axes.to_pixels(np.array([x_lo, x_lo, x_hi, x_hi]), np.array([y_lo, y_hi, y_hi, y_lo]))


def _truth(adata, layer, x, y, box):
    import cytopy

    values = np.asarray(adata.layers[layer])
    xi, yi = cytopy.channel_index(adata, x), cytopy.channel_index(adata, y)
    return (
        (values[:, xi] > box[0])
        & (values[:, xi] < box[1])
        & (values[:, yi] > box[2])
        & (values[:, yi] < box[3])
    )


def test_a_child_gate_counts_only_what_was_drawn_for_it(cv):
    """A leftover shape used to be OR-ed into the next gate, inflating it."""
    parent_box = (2.0, 9.0, -2.0, 9.0)
    cv.gates.add_polygons([_rect(cv, *parent_box)])
    cv.apply_gate("parent")
    assert len(cv.gates.data) == 0  # committed, so the canvas clears

    child_box = (3.0, 5.0, 0.0, 2.0)
    cv.w_parent.value = "parent"
    cv.gates.add_polygons([_rect(cv, *child_box)])
    cv.apply_gate("child", parent="parent")

    expected = _truth(cv.adata, "asinh", "CD3", "CD19", parent_box) & _truth(
        cv.adata, "asinh", "CD3", "CD19", child_box
    )
    assert int(cv.adata.obs["child"].sum()) == int(expected.sum())
    assert np.array_equal(cv.adata.obs["child"].to_numpy(), expected)
    # and the child is a strict subset of its parent
    assert not (cv.adata.obs["child"] & ~cv.adata.obs["parent"]).any()


def test_two_shapes_still_make_one_gate(cv):
    """Clearing on apply must not break a gate deliberately drawn in two parts."""
    left, right = (2.0, 3.0, -2.0, 9.0), (5.0, 6.0, -2.0, 9.0)
    cv.gates.add_polygons([_rect(cv, *left)])
    cv.gates.add_polygons([_rect(cv, *right)])
    cv.apply_gate("both")
    expected = _truth(cv.adata, "asinh", "CD3", "CD19", left) | _truth(
        cv.adata, "asinh", "CD3", "CD19", right
    )
    assert int(cv.adata.obs["both"].sum()) == int(expected.sum())


@pytest.mark.parametrize(
    ("change", "value"),
    [("w_bins", 128), ("w_parent", "seed"), ("w_robust", False), ("w_y", "CD8 (APC-A)")],
)
def test_a_shape_keeps_its_data_meaning_when_the_axes_move(cv, change, value):
    """The shape lives in pixels but means data; a rescale must not shift it."""
    cv.gates.add_polygons([_rect(cv, 2.0, 9.0, -2.0, 9.0)])
    cv.apply_gate("seed")

    cv.gates.add_polygons([_rect(cv, 3.0, 5.0, 0.0, 2.0)])
    before = cv.axes.to_display(np.asarray(cv.gates.data[0], dtype=float))
    getattr(cv, change).value = value
    after = cv.axes.to_display(np.asarray(cv.gates.data[0], dtype=float))
    assert np.abs(after - before).max() < 1e-6


def test_shape_types_survive_being_reprojected(cv):
    cv.gates.add_rectangles([_rect(cv, 2.0, 4.0, 0.0, 2.0)])
    cv.gates.add_ellipses([_rect(cv, 5.0, 6.0, 1.0, 3.0)])
    kinds = [str(k) for k in cv.gates.shape_type]
    cv.w_bins.value = 128
    assert [str(k) for k in cv.gates.shape_type] == kinds
    assert len(cv.gates.data) == 2


def test_events_off_the_axes_are_reported_not_hidden(cv):
    """Robust limits clip the axes, so a few events are drawn nowhere."""
    assert cv.w_robust.value is True
    off = cv.off_axis_count()
    assert off > 0

    cv.gates.add_polygons([_rect(cv, 2.0, 9.0, -2.0, 9.0)])
    cv.apply_gate("g")
    assert f"{off:,} outside the axes" in cv.w_status.value

    cv.w_robust.value = False
    assert cv.off_axis_count() == 0  # the full range holds everything


def test_apply_gate_needs_a_shape(cv):
    with pytest.raises(ValueError, match="draw a shape"):
        cv.apply_gate("nothing")
    assert "nothing" not in cv.adata.obs


@pytest.fixture
def pair(demo, demo_path, make_napari_viewer):
    """Two samples in one viewer, the second a quarter as bright."""
    import cytopy
    from cytopy.viewer import CytoViewer

    cytopy.asinh_transform(demo, 150.0)
    dim = cytopy.read_fcs(demo_path, sample_id="dim")
    dim.X = dim.X * np.float32(0.4)
    cytopy.asinh_transform(dim, 150.0)
    return CytoViewer(
        [demo, dim], layer="asinh", x="CD3", y="CD19", bins=128, viewer=make_napari_viewer()
    )


# --------------------------------------------------------------------------
# the colour bar
# --------------------------------------------------------------------------
def test_the_colour_bar_is_drawn_by_default(cv):
    assert cv.w_colorbar.value is True
    assert cv.colorbar.visible is True
    assert cv.colorbar.data.shape[0] == cv.bins
    # It sits to the right of the plot, and shares its scale.
    assert cv.colorbar.translate[1] > cv.bins
    assert cv.colorbar.contrast_limits == cv.density.contrast_limits


def test_the_bar_runs_bright_at_the_top(cv):
    column = np.asarray(cv.colorbar.data)[:, 0]
    assert column[0] == pytest.approx(cv.density.contrast_limits[1])
    assert column[-1] == 0.0


def test_the_bar_is_labelled_as_a_share_of_the_peak(cv):
    """Events per bin is a property of the grid, not the data.

    The same events in the same channels peak at ~470 per bin at 64 bins and
    ~5 at 1024, so labelling the bar with counts made the numbers lurch about
    whenever the bins or the channels changed. A share of the peak holds still.
    """
    labels = [str(t) for t in cv.labels.text.string.array]
    for tick in ("0%", "25%", "50%", "75%", "100%"):
        assert tick in labels
    assert "of peak" in labels

    peak = cv.peak_density()
    assert peak > 0
    assert f"peak {peak:,.0f}" in cv.w_status.value

    # The ticks hold still across a change that moves the peak a long way.
    cv.w_bins.value = 64
    assert cv.peak_density() > peak * 2
    assert all(t in [str(x) for x in cv.labels.text.string.array] for t in ("0%", "100%"))


def test_the_peak_follows_the_data_not_the_labels(cv):
    before = cv.peak_density()
    cv.w_x.value = "CD8 (APC-A)"
    after = cv.peak_density()
    assert before != after  # the absolute number does move ...
    labels = [str(t) for t in cv.labels.text.string.array]
    assert "100%" in labels and "of peak" in labels  # ... the bar does not


def test_the_colour_bar_can_be_turned_off(cv):
    cv.w_colorbar.value = False
    assert cv.colorbar.visible is False
    labels = [str(t) for t in cv.labels.text.string.array]
    assert "smoothed" not in labels and "events/bin" not in labels


def test_the_bar_moves_past_the_last_panel(pair):
    alone = pair.colorbar.translate[1]
    pair.add_panel("density")
    assert pair.colorbar.translate[1] > alone


# --------------------------------------------------------------------------
# the 1-D histogram
# --------------------------------------------------------------------------
def test_histogram_swaps_the_density_out(pair):
    pair.w_plot.value = "histogram"
    assert pair.histogram is True
    assert pair.curves.visible is True
    assert pair.density.visible is False
    assert pair.colorbar.visible is False

    pair.w_plot.value = "density"
    assert pair.density.visible is True
    assert pair.curves.visible is False


def test_one_curve_per_sample(pair):
    pair.w_plot.value = "histogram"
    groups = pair.histogram_groups()
    assert [name for name, _ in groups] == ["demo", "dim"]
    assert all(int(mask.sum()) == 60_000 for _, mask in groups)
    # two curves, plus a legend rule for each
    assert len(pair.curves.data) == 4
    labels = [str(t) for t in pair.labels.text.string.array]
    assert "demo (60,000)" in labels and "dim (60,000)" in labels
    assert "2 curve(s)" in pair.w_status.value


def test_the_curves_follow_the_data(pair):
    """The dim sample is 0.4x, so its distribution sits to the left."""
    pair.w_plot.value = "histogram"

    def peak_x(index):
        path = np.asarray(pair.curves.data[index])
        column = path[np.argmin(path[:, 0]), 1]  # row 0 is the top of the plot
        return float(pair.axes.to_display(np.array([[0.0, column]]))[0, 0])

    assert peak_x(1) < peak_x(0)


def test_the_sample_selector_narrows_to_one_curve(pair):
    pair.w_plot.value = "histogram"
    pair.w_sample.value = "dim"
    assert [name for name, _ in pair.histogram_groups()] == ["dim"]
    assert len(pair.curves.data) == 2  # one curve, one legend rule


def test_the_parent_gate_applies_to_the_curves(cv):
    cv.gates.add_polygons([_rect(cv, 3.0, 9.0, -2.0, 9.0)])
    cv.apply_gate("bright")
    n = int(cv.adata.obs["bright"].sum())

    cv.w_plot.value = "histogram"
    cv.w_parent.value = "bright"
    assert int(cv.histogram_groups()[0][1].sum()) == n


def test_the_y_axis_is_a_density_not_a_channel(pair):
    pair.w_plot.value = "histogram"
    labels = [str(t) for t in pair.labels.text.string.array]
    assert "density" in labels
    assert "CD19 (PE-A)" not in labels  # the y channel means nothing here
    assert "CD3 (FITC-A)" in labels  # ... but the x channel still does

    pair.w_peak.value = True
    assert "peak" in [str(t) for t in pair.labels.text.string.array]


def test_curves_are_densities_so_samples_compare(pair):
    """Unit area, so a bigger sample does not simply draw a taller curve."""
    from cytopy.density import density_curve

    pair.w_plot.value = "histogram"
    _, mask = pair.histogram_groups()[0]
    values = pair._column(pair.w_x.value, mask)
    curve = density_curve(values, pair.axes.x_lo, pair.axes.x_hi, pair.bins)
    width = (pair.axes.x_hi - pair.axes.x_lo) / pair.bins
    assert curve.sum() * width == pytest.approx(1.0, abs=0.02)


def test_a_shape_gates_an_interval_on_a_histogram(cv):
    """There is no second channel to be inside of, so a box means an x range."""
    import cytopy

    cv.w_plot.value = "histogram"
    cv.gates.add_rectangles([_rect(cv, 3.0, 5.0, 0.0, 1.0)])
    mask = cv.current_gate_mask()

    values = np.asarray(cv.adata.layers["asinh"])[:, cytopy.channel_index(cv.adata, "CD3")]
    expected = (values >= 3.0) & (values <= 5.0)
    assert int(mask.sum()) == int(expected.sum())
    assert np.array_equal(mask, expected)


def test_settings_that_do_not_apply_are_greyed_out(pair):
    pair.w_plot.value = "histogram"
    assert pair.w_y.enabled is False
    assert pair.w_colorbar.enabled is False
    assert pair.w_peak.enabled is True

    pair.w_plot.value = "density"
    assert pair.w_y.enabled is True
    assert pair.w_peak.enabled is False


def test_an_empty_selection_draws_nothing(cv):
    cv.gates.add_polygons([_rect(cv, 100.0, 101.0, 100.0, 101.0)])
    cv.apply_gate("empty")
    cv.w_plot.value = "histogram"
    cv.w_parent.value = "empty"
    assert cv.histogram_groups() == []
    assert len(cv.curves.data) == 0


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------
def test_a_viewer_starts_with_one_panel(cv):
    assert len(cv.panels) == 1
    assert cv.active == 0
    assert cv.panel is cv.panels[0]
    assert cv.density is cv.panel.image  # the old name still points at the active one
    assert cv.axes is cv.panel.axes


def test_each_panel_is_its_own_napari_layer(cv):
    """What duplicating the density layer was reaching for."""
    cv.add_panel("density")
    names = [layer.name for layer in cv.viewer.layers]
    assert "panel 1" in names and "panel 2" in names
    assert cv.panels[0].image is not cv.panels[1].image
    # hiding one leaves the other alone
    cv.panels[0].image.visible = False
    cv.refresh()
    assert cv.panels[1].image.visible is True


def test_panels_can_be_density_or_histogram(cv):
    cv.add_panel("histogram", x="CD8 (APC-A)")
    assert [p.kind for p in cv.panels] == ["density", "histogram"]
    assert cv.panels[0].image.visible and not cv.panels[0].curves.visible
    assert cv.panels[1].curves.visible and not cv.panels[1].image.visible


def test_a_new_panel_copies_the_active_one_then_diverges(cv):
    cv.w_x.value = "CD8 (APC-A)"
    cv.add_panel("density")
    assert cv.panels[1].x == "CD8 (APC-A)"  # starts from what you were looking at
    cv.w_y.value = "FSC-A"
    assert cv.panels[1].y == "FSC-A"
    assert cv.panels[0].y == "CD19 (PE-A)"  # ... and does not drag the first along


def test_clicking_a_panel_rebinds_the_settings(cv):
    cv.add_panel("density", x="CD8 (APC-A)", y="FSC-A")
    assert cv.active == 1
    first = cv.panels[0]

    assert cv._panel_at(first.row + 10, first.col + 10) == 0
    cv.select_panel(0)
    assert (cv.w_x.value, cv.w_y.value) == ("CD3 (FITC-A)", "CD19 (PE-A)")
    cv.select_panel(1)
    assert (cv.w_x.value, cv.w_y.value) == ("CD8 (APC-A)", "FSC-A")
    assert cv.w_panel.value == "2"


def test_clicking_outside_every_panel_changes_nothing(cv):
    cv.add_panel("density")
    assert cv._panel_at(-10_000, -10_000) is None


def test_display_settings_apply_to_every_panel(cv):
    """bins, colormap and the rest are global; the plot settings are not."""
    cv.add_panel("density", x="CD8 (APC-A)")
    cv.w_bins.value = 64
    assert all(p.axes.bins == 64 for p in cv.panels)
    assert all(np.asarray(p.image.data).shape == (64, 64) for p in cv.panels)


def test_panels_share_one_intensity_scale(cv):
    cv.add_panel("density", x="CD8 (APC-A)", y="FSC-A")
    limits = [tuple(p.image.contrast_limits) for p in cv.panels]
    assert limits[0] == limits[1]


def test_arrange_tiles_the_panels(cv):
    for _ in range(3):
        cv.add_panel("density")
    cv.panels[1].row += 5_000
    cv.arrange()
    positions = [(round(p.row), round(p.col)) for p in cv.panels]
    assert len(set(positions)) == 4  # no two on top of each other
    assert positions[0] == (0, 0)
    step = round(cv.bins * 1.12)
    assert positions[1] == (0, step)


def test_a_panel_can_be_moved_and_its_box_follows(cv):
    cv.add_panel("density")
    panel = cv.panels[1]
    panel.row, panel.col = 500.0, 700.0
    cv.refresh()
    assert tuple(panel.image.translate) == (500.0, 700.0)
    # the frame is redrawn around the new position
    frame = np.vstack([np.asarray(line) for line in cv.grid.data])
    assert frame[:, 0].max() >= 500.0
    assert frame[:, 1].max() >= 700.0


def test_panels_stacked_on_one_spot_overlay_with_the_denser_winning(cv):
    """Drag one panel onto another and neither should bury the other."""
    cv.add_panel("density", sample="<all>")
    cv.panels[1].row, cv.panels[1].col = cv.panels[0].row, cv.panels[0].col
    cv.w_x.value = "CD8 (APC-A)"  # different data, same place
    cv.refresh()

    a = np.asarray(cv.panels[0].image.data)
    b = np.asarray(cv.panels[1].image.data)
    assert not ((a > 0) & (b > 0)).any()
    assert (a > 0).any() and (b > 0).any()


def test_the_last_panel_cannot_be_removed(cv):
    with pytest.raises(ValueError, match="at least one panel"):
        cv.remove_panel()
    cv.add_panel("density")
    cv.remove_panel()
    assert len(cv.panels) == 1
    assert "panel 2" not in [layer.name for layer in cv.viewer.layers]


def test_gates_are_drawn_against_the_active_panel(cv):
    cv.add_panel("density", x="CD8 (APC-A)", y="FSC-A")
    cv.select_panel(0)
    cv.gates.add_polygons([_rect(cv, 3.0, 9.0, -2.0, 9.0)])
    cv.apply_gate("on panel 1")

    expected = _truth(cv.adata, "asinh", "CD3", "CD19", (3.0, 9.0, -2.0, 9.0))
    assert int(cv.adata.obs["on panel 1"].sum()) == int(expected.sum())
