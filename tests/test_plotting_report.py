"""Static figures, the filter log, and the report that assembles them."""

import re

import numpy as np
import pandas as pd
import pytest

import cytopy


@pytest.fixture(autouse=True)
def _agg():
    import matplotlib

    matplotlib.use("Agg", force=True)


# --------------------------------------------------------------------------
# biaxial plots
# --------------------------------------------------------------------------
def test_biaxial_is_a_density_by_default(demo):
    ax = cytopy.plot_biaxial(demo, "CD3", "CD19")
    assert len(ax.images) == 1 and not ax.collections
    image = ax.images[0].get_array()
    assert image.shape == (512, 512) and image.max() > 0
    assert ax.get_xlabel() == "CD3 (FITC-A)" and ax.get_ylabel() == "CD19 (PE-A)"
    assert "60,000 events" in ax.get_title(loc="left")


def test_biaxial_colours_by_a_boolean_column(demo):
    demo.obs["big"] = np.asarray(demo.X[:, 0]) > np.median(demo.X[:, 0])
    ax = cytopy.plot_biaxial(demo, "CD3", "CD19", color_by="big")
    assert len(ax.images) == 1 and len(ax.collections) == 1
    assert ax.images[0].get_cmap().name == "Greys"  # the highlight carries the colour
    assert ax.get_legend() is not None
    with pytest.raises(KeyError, match="to colour by"):
        cytopy.plot_biaxial(demo, "CD3", "CD19", color_by="nope")


def test_biaxial_subsets(demo):
    demo.obs["half"] = np.arange(demo.n_obs) < demo.n_obs // 2
    ax = cytopy.plot_biaxial(demo, "CD3", "CD19", subset="half")
    assert "30,000 events" in ax.get_title(loc="left")
    ax = cytopy.plot_biaxial(demo, "CD3", "CD19", subset=np.zeros(demo.n_obs, dtype=bool))
    assert "0 events" in ax.get_title(loc="left")
    with pytest.raises(ValueError, match="expected 60000"):
        cytopy.plot_biaxial(demo, "CD3", "CD19", subset=np.ones(3, dtype=bool))


def test_biaxial_axes_read_in_raw_units(demo):
    """The viewer's rule: a layer that records its transform gets decade ticks."""
    cytopy.asinh_transform(demo, 150.0)
    plain = cytopy.plot_biaxial(demo, "CD3", "CD19")
    labelled = cytopy.plot_biaxial(demo, "CD3", "CD19", layer="asinh")
    assert not any(
        "¹⁰" in t.get_text() or "10" in t.get_text() for t in plain.get_xticklabels()[:1]
    )
    assert any(t.get_text() == "0" for t in labelled.get_xticklabels())
    assert any("10" in t.get_text() for t in labelled.get_xticklabels())


def test_biaxial_cofactor_transforms_for_display_only(demo):
    before = np.asarray(demo.X).copy()
    ax = cytopy.plot_biaxial(demo, "CD3", "CD19", cofactor=5.0)
    assert np.array_equal(np.asarray(demo.X), before)  # nothing written back
    assert "asinh" not in demo.layers
    assert ax.get_xlim()[1] < 20  # plotted on the compressed scale
    assert any("10" in t.get_text() for t in ax.get_xticklabels())


def test_biaxial_draws_rectangles(demo):
    ax = cytopy.plot_biaxial(demo, "CD3", "CD19", rectangles=[(100.0, 500.0, 100.0, 500.0)])
    dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert len(dashed) == 1
    # The outline must not drag the axes with it.
    assert ax.get_xlim()[0] < 100.0


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
def _draw_gate(adata, name="lymphs"):
    x = np.asarray(adata.X[:, cytopy.channel_index(adata, "CD3")], dtype=float)
    y = np.asarray(adata.X[:, cytopy.channel_index(adata, "CD19")], dtype=float)
    verts = [[500.0, -200.0], [500.0, 2000.0], [20000.0, 2000.0], [20000.0, -200.0]]
    mask = cytopy.polygon_mask(np.column_stack([x, y]), np.asarray(verts))
    cytopy.add_gate(
        adata,
        name,
        mask,
        meta={"x": "CD3 (FITC-A)", "y": "CD19 (PE-A)", "layer": "X", "vertices": [verts]},
    )
    return name


def test_plot_gate_redraws_where_it_was_drawn(demo):
    name = _draw_gate(demo)
    ax = cytopy.plot_gate(demo, name)
    assert ax.get_xlabel() == "CD3 (FITC-A)"
    assert any(ln.get_linestyle() == "--" for ln in ax.lines)
    assert len(ax.collections) == 1  # the gated events, picked out
    assert name in ax.get_title(loc="left")


def test_plot_gate_needs_a_recorded_gate(demo):
    with pytest.raises(KeyError, match="no gate"):
        cytopy.plot_gate(demo, "never drawn")
    cytopy.add_gate(demo, "bare", np.ones(demo.n_obs, dtype=bool))
    with pytest.raises(KeyError, match="did not record which channels"):
        cytopy.plot_gate(demo, "bare")


def test_a_gate_outline_is_skipped_on_the_wrong_axes(demo):
    name = _draw_gate(demo)
    ax = cytopy.plot_biaxial(demo, "CD3", "CD8", gate=name)
    assert not [ln for ln in ax.lines if ln.get_linestyle() == "--"]


# --------------------------------------------------------------------------
# the filter log
# --------------------------------------------------------------------------
def test_filter_events_records_what_went(demo):
    keep = np.arange(demo.n_obs) % 2 == 0
    out = cytopy.filter_events(demo, keep, step="every other", reason="for the sake of it")
    assert out.n_obs == 30_000
    log = cytopy.filter_log(out)
    assert list(log["step"]) == ["every other"]
    assert log.iloc[0].to_dict() == {
        "step": "every other",
        "reason": "for the sake of it",
        "n_before": 60_000,
        "n_removed": 30_000,
        "pct_removed": 50.0,
        "n_after": 30_000,
    }


def test_the_log_accumulates_and_survives_filtering(demo):
    first = cytopy.filter_events(demo, np.arange(demo.n_obs) < 40_000, step="one")
    second = cytopy.filter_events(first, np.arange(first.n_obs) < 10_000, step="two")
    log = cytopy.filter_log(second)
    assert list(log["step"]) == ["one", "two"]
    assert list(log["n_before"]) == [60_000, 40_000]
    assert list(log["n_after"]) == [40_000, 10_000]


def test_record_filter_counts_without_removing(demo):
    entry = cytopy.record_filter(demo, "marked", np.ones(demo.n_obs, dtype=bool), reason="none")
    assert entry["n_removed"] == 0 and demo.n_obs == 60_000
    with pytest.raises(ValueError, match="expected 60000"):
        cytopy.record_filter(demo, "bad", np.ones(5, dtype=bool))


def test_an_empty_log_is_an_empty_table(demo):
    assert cytopy.filter_log(demo).empty
    assert "step" in cytopy.filter_log(demo).columns


def test_remove_beads_records_the_gate(cytof):
    cytopy.gate_beads(cytof)
    n_beads = int(cytof.obs["bead"].sum())
    clean = cytopy.remove_beads(cytof)
    assert clean.n_obs == cytof.n_obs - n_beads
    row = cytopy.filter_log(clean).iloc[0]
    assert row["step"] == "remove beads" and row["n_removed"] == n_beads
    assert f"{n_beads:,}" in row["reason"]


def test_remove_beads_can_also_take_the_bead_adjacent(barcoded):
    """The distance rule catches what sits next to the beads but outside the gate."""
    mask = cytopy.gate_beads(barcoded)
    # Let a hundred beads slip the gate, as a bead-cell doublet would.
    slipped = np.flatnonzero(mask)[:100]
    barcoded.obs.loc[barcoded.obs_names[slipped], "bead"] = False

    plain = cytopy.remove_beads(barcoded)
    assert plain.n_obs == barcoded.n_obs - (int(mask.sum()) - 100)

    strict = cytopy.remove_beads(barcoded, distance_cutoff=10.0)
    assert strict.n_obs == barcoded.n_obs - int(mask.sum())  # all of them, back again
    # Two calls on the same object, so the log has an entry for each.
    log = cytopy.filter_log(strict)
    assert list(log["step"]) == ["remove beads", "remove beads"]
    assert "100 more within 10 of the bead centroid" in log.iloc[-1]["reason"]
    with pytest.raises(KeyError, match="gate_beads"):
        cytopy.remove_beads(barcoded, bead_key="nope")


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
def test_report_is_one_self_contained_file(barcoded, tmp_path):
    cytopy.gate_beads(barcoded)
    cytopy.normalise_beads(barcoded)
    gated = barcoded.copy()
    clean = cytopy.remove_beads(barcoded, distance_cutoff=3.0, layer="normalised")
    cytopy.debarcode(clean, "pd20", layer="normalised")

    path = cytopy.report(clean, tmp_path / "qc.html", title="Run 1", beads=gated, max_barcodes=3)
    assert path.exists() and list(tmp_path.iterdir()) == [path]
    text = path.read_text()
    assert text.startswith("<!doctype html>")
    assert "Run 1" in text
    assert re.findall(r"<h2>(.*?)</h2>", text) == [
        "Summary",
        "What was removed",
        "Bead normalisation",
        "Debarcoding",
    ]
    # Every figure is inlined, so there is nothing to lose alongside the file.
    assert len(re.findall(r"data:image/png;base64,", text)) >= 4
    assert 'src="http' not in text and "<link" not in text
    # The filter log is in there as numbers, not just pictures.
    assert "remove beads" in text and "debarcode" in text


def test_report_skips_sections_with_nothing_to_show(demo, tmp_path):
    text = cytopy.report(demo, tmp_path / "plain.html").read_text()
    assert re.findall(r"<h2>(.*?)</h2>", text) == ["Summary"]
    assert "60,000 events" in text


def test_report_sections_can_be_chosen(cytof, tmp_path):
    cytopy.gate_beads(cytof)
    cytopy.normalise_beads(cytof)
    text = cytopy.report(cytof, tmp_path / "beads.html", sections=["beads"]).read_text()
    assert re.findall(r"<h2>(.*?)</h2>", text) == ["Bead normalisation"]
    with pytest.raises(ValueError, match="unknown report section"):
        cytopy.report(cytof, tmp_path / "x.html", sections=["nope"])


def test_the_bead_figure_survives_the_beads_being_removed(cytof, tmp_path):
    """The curves are stashed at normalisation, so the report still has them."""
    cytopy.gate_beads(cytof)
    cytopy.normalise_beads(cytof)
    clean = cytopy.remove_beads(cytof)
    assert not clean.obs["bead"].any()

    fig = cytopy.plot_beads_over_time(clean)
    assert [ax.get_title(loc="left") for ax in fig.axes] == ["before", "after"]
    text = cytopy.report(clean, tmp_path / "qc.html", sections=["beads"]).read_text()
    assert "data:image/png;base64," in text
    assert "cannot be redrawn" in text  # honest about the gate panels it cannot draw


def test_without_stashed_curves_the_figure_is_refused(cytof):
    cytopy.gate_beads(cytof)
    cytopy.normalise_beads(cytof, stash_curves=None)
    clean = cytopy.remove_beads(cytof)
    with pytest.raises(KeyError, match="no stashed curves"):
        cytopy.plot_beads_over_time(clean)


def test_report_shows_gates(demo, tmp_path):
    _draw_gate(demo)
    text = cytopy.report(demo, tmp_path / "gates.html").read_text()
    assert "Gates" in re.findall(r"<h2>(.*?)</h2>", text)
    assert "lymphs" in text
    assert "data:image/png;base64," in text


def test_everything_survives_write_h5ad(barcoded, tmp_path):
    """A whole processed run has to save and come back -- log, gates and all."""
    import anndata

    cytopy.gate_beads(barcoded)
    cytopy.normalise_beads(barcoded)
    clean = cytopy.remove_beads(barcoded, distance_cutoff=5.0, layer="normalised")
    cytopy.debarcode(clean, "pd20", layer="normalised")

    path = tmp_path / "run.h5ad"
    clean.write_h5ad(path)
    back = anndata.read_h5ad(path)

    assert back.n_obs == clean.n_obs
    pd.testing.assert_frame_equal(cytopy.filter_log(back), cytopy.filter_log(clean))
    assert list(back.obs["bc_id"]) == list(clean.obs["bc_id"])
    assert back.uns["cytopy"]["beads"]["baseline"] == clean.uns["cytopy"]["beads"]["baseline"]
    # uns lists come back as arrays, hence the list() on both sides.
    assert list(back.uns["cytopy"]["debarcode"]["channels"]) == list(
        clean.uns["cytopy"]["debarcode"]["channels"]
    )

    # The restored object is enough to write the report from.
    text = cytopy.report(back, tmp_path / "qc.html", max_barcodes=2).read_text()
    assert "remove beads" in text and "Debarcoding" in text


def test_the_log_keeps_appending_after_a_round_trip(barcoded, tmp_path):
    import anndata

    cytopy.gate_beads(barcoded)
    first = cytopy.remove_beads(barcoded)
    first.write_h5ad(tmp_path / "one.h5ad")

    back = anndata.read_h5ad(tmp_path / "one.h5ad")
    second = cytopy.filter_events(back, np.arange(back.n_obs) < 1000, step="later")
    log = cytopy.filter_log(second)
    assert list(log["step"]) == ["remove beads", "later"]
    assert list(log["n_after"]) == [first.n_obs, 1000]


# --------------------------------------------------------------------------
# the gating hierarchy as a PDF
# --------------------------------------------------------------------------
def _hierarchy(adata):
    """Three nested gates, the last drawn on a different pair of channels."""
    import cytopy

    cytopy.asinh_transform(adata, 150.0)
    values = np.asarray(adata.layers["asinh"])

    def gate(name, x, y, box, parent=None):
        xi, yi = cytopy.channel_index(adata, x), cytopy.channel_index(adata, y)
        mask = (
            (values[:, xi] > box[0])
            & (values[:, xi] < box[1])
            & (values[:, yi] > box[2])
            & (values[:, yi] < box[3])
        )
        cytopy.add_gate(
            adata,
            name,
            mask,
            parent=parent,
            meta={
                "x": str(adata.var_names[xi]),
                "y": str(adata.var_names[yi]),
                "layer": "asinh",
                "vertices": [
                    [[box[0], box[2]], [box[0], box[3]], [box[1], box[3]], [box[1], box[2]]]
                ],
                "shape_types": ["polygon"],
            },
        )

    gate("lymphocytes", "CD3", "CD19", (2.0, 9.0, -2.0, 9.0))
    gate("T cells", "CD3", "CD19", (3.0, 9.0, -2.0, 4.0), parent="lymphocytes")
    gate("CD8+", "CD8", "CD3", (3.0, 9.0, 3.0, 9.0), parent="T cells")
    return adata


def test_gating_pdf_writes_a_page_per_gate(demo, tmp_path):
    _hierarchy(demo)
    path = cytopy.gating_pdf(demo, tmp_path / "gating.pdf", ncols=3, per_page=3)
    assert path.exists() and path.stat().st_size > 10_000
    assert path.read_bytes().startswith(b"%PDF")
    # a contents page plus one page of three plots
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: F401

    assert path.read_bytes().count(b"/Page") >= 2


def test_the_hierarchy_comes_out_parents_first(demo):
    from cytopy.report import _hierarchy as order_of

    _hierarchy(demo)
    gates = demo.uns["cytopy"]["gates"]
    assert order_of(gates) == ["lymphocytes", "T cells", "CD8+"]


def test_each_gate_is_drawn_on_the_plane_it_was_drawn_in(demo):
    import matplotlib.pyplot as plt

    from cytopy.plotting import plot_biaxial
    from cytopy.report import _draw_gate_page

    _hierarchy(demo)
    gates = demo.uns["cytopy"]["gates"]
    _, axes = plt.subplots(1, 3)
    for ax, name in zip(axes, ["lymphocytes", "T cells", "CD8+"]):
        _draw_gate_page(demo, demo, gates, name, ax, plot_biaxial, {})

    # the last gate was drawn on a different pair, and follows it
    assert axes[0].get_xlabel() == "CD3 (FITC-A)"
    assert axes[2].get_xlabel() == "CD8 (APC-A)"
    assert axes[2].get_ylabel() == "CD3 (FITC-A)"
    # each title carries its lineage and its share of its own parent
    n = int(demo.obs["T cells"].sum())
    total = int(demo.obs["lymphocytes"].sum())
    title = axes[1].get_title(loc="left")
    assert "lymphocytes > T cells" in title
    assert f"{n:,} of {total:,}" in title and f"{100 * n / total:.1f}%" in title


def test_the_contents_page_lists_the_tree(demo):
    from cytopy.report import _hierarchy as order_of
    from cytopy.report import _hierarchy_page

    _hierarchy(demo)
    gates = demo.uns["cytopy"]["gates"]
    fig = _hierarchy_page(demo, gates, order_of(gates), "run 1")
    text = [t.get_text() for t in fig.axes[0].texts]
    assert "run 1" in text
    assert any("60,000 events" in t for t in text)

    cells = [c.get_text().get_text() for c in fig.axes[0].tables[0].get_celld().values()]
    assert any(c.strip() == "lymphocytes" for c in cells)
    assert any(c.startswith("    ") and c.strip() == "T cells" for c in cells)  # indented
    assert f"{int(demo.obs['CD8+'].sum()):,}" in cells


def test_gating_pdf_needs_something_to_draw(demo, tmp_path):
    with pytest.raises(ValueError, match="no gates with an outline"):
        cytopy.gating_pdf(demo, tmp_path / "empty.pdf")

    cytopy.add_gate(demo, "from a mask", np.ones(demo.n_obs, dtype=bool))
    with pytest.raises(ValueError, match="no gates with an outline"):
        cytopy.gating_pdf(demo, tmp_path / "empty.pdf")
