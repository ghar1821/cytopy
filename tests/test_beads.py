"""Bead normalisation (Finck et al.), checked against CATALYST and premessa.

The expected values in ``test_matches_the_r_reference_implementations`` were
produced by running the reference R code -- premessa's ``identify_beads`` /
``correct_data_channels`` and CATALYST's ``normCytof`` arithmetic -- verbatim
on this same synthetic file. At float64 the two agree to 1e-12; the tolerance
here is loosened only for the float32 the layer is stored as.
"""

import numpy as np
import pandas as pd
import pytest

import cytopy
from cytopy.beads import _runmed

BEADS = ["Ce140Di", "Eu151Di", "Eu153Di", "Ho165Di", "Lu175Di"]


# --------------------------------------------------------------------------
# finding the channels
# --------------------------------------------------------------------------
def test_bead_channels_default_to_the_fluidigm_set(cytof):
    assert cytopy.bead_channels(cytof) == BEADS
    assert cytopy.bead_channels(cytof, "dvs") == BEADS
    assert cytopy.dna_channel(cytof) == "DNA2 (Ir193Di)"
    assert cytopy.dna_channel(cytof, 191) == "DNA1 (Ir191Di)"


def test_bead_channels_accept_isotopes_and_masses(cytof):
    assert cytopy.bead_channels(cytof, [140, 151]) == ["Ce140Di", "Eu151Di"]
    assert cytopy.bead_channels(cytof, ["Ce140", "140Ce"]) == ["Ce140Di"]
    assert cytopy.bead_channels(cytof, ["Ho165Di"]) == ["Ho165Di"]


def test_a_missing_bead_channel_is_fatal(cytof):
    """A bead set silently missing a channel still normalises, just wrongly."""
    with pytest.raises(KeyError, match="La139"):
        cytopy.bead_channels(cytof, "beta")
    with pytest.raises(KeyError, match="unknown bead set"):
        cytopy.bead_channels(cytof, "nonesuch")


def test_mass_channels_leave_the_housekeeping_columns_alone(cytof):
    scaled = cytopy.mass_channels(cytof)
    assert "Time" not in scaled and "Event_length" not in scaled
    assert set(BEADS) <= set(scaled)
    assert cytopy.mass_channels(cytof, ["Ce140Di"]) == ["Ce140Di"]


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
def test_default_gate_is_premessas(cytof):
    gates = cytopy.bead_gates(cytof)
    assert set(gates) == set(BEADS)
    assert all(g == {"x": (2.0, 5.0), "y": (-1.0, 2.0)} for g in gates.values())


def test_gate_overrides_merge_per_axis(cytof):
    gates = cytopy.bead_gates(cytof, gates={"Ce140Di": {"x": (2.5, 4.5)}})
    assert gates["Ce140Di"] == {"x": (2.5, 4.5), "y": (-1.0, 2.0)}
    assert gates["Eu151Di"] == {"x": (2.0, 5.0), "y": (-1.0, 2.0)}
    with pytest.raises(KeyError, match="not a bead channel"):
        cytopy.bead_gates(cytof, gates={"DNA1": {"x": (0, 1)}})


def test_gate_beads_writes_a_mask_and_records_the_gate(cytof):
    mask = cytopy.gate_beads(cytof)
    assert mask.dtype == bool and mask.sum() > 100
    assert np.array_equal(np.asarray(cytof.obs["bead"]), mask)
    info = cytof.uns["cytopy"]["beads"]
    assert info["channels"] == BEADS
    assert info["n_beads"] == int(mask.sum())
    assert info["gates"]["Ce140Di"] == {"x": [2.0, 5.0], "y": [-1.0, 2.0]}


def test_a_tighter_gate_selects_fewer_beads(cytof):
    wide = cytopy.gate_beads(cytof, key_added=None)
    tight = cytopy.gate_beads(cytof, gates={"Ce140Di": {"x": (4.0, 5.0)}}, key_added=None)
    assert 0 < tight.sum() < wide.sum()
    assert not tight[~wide].any()  # tightening only ever removes events


def test_trim_drops_the_outlying_beads(cytof):
    plain = cytopy.gate_beads(cytof, key_added=None)
    trimmed = cytopy.gate_beads(cytof, trim=1.0, key_added=None)
    assert trimmed.sum() < plain.sum()
    assert not trimmed[~plain].any()


def test_a_gate_that_holds_nothing_is_an_error(cytof):
    with pytest.raises(ValueError, match="selects no events"):
        cytopy.gate_beads(cytof, gates={"Ce140Di": {"x": (19.0, 20.0)}})


def test_a_gate_drawn_elsewhere_is_used_as_is(cytof):
    """A boolean obs column -- e.g. a polygon drawn in the viewer -- wins."""
    mask = cytopy.gate_beads(cytof, key_added=None)
    mask[np.flatnonzero(mask)[:50]] = False
    cytof.obs["bead"] = mask
    cytopy.normalise_beads(cytof)
    assert cytof.uns["cytopy"]["beads"]["samples"]["demo_cytof"]["n_beads"] == int(mask.sum())


# --------------------------------------------------------------------------
# the running median
# --------------------------------------------------------------------------
def test_runmed_matches_r():
    """Values from R: runmed(x, k, endrule) on 1..9 with an outlier."""
    x = np.array([1.0, 2, 3, 40, 5, 6, 7, 8, 9])
    assert np.allclose(_runmed(x, 3, "keep"), [1, 2, 3, 5, 6, 6, 7, 8, 9])
    assert np.allclose(_runmed(x, 3, "constant"), [2, 2, 3, 5, 6, 6, 7, 8, 8])
    assert np.allclose(_runmed(x, 3, "median"), [1, 2, 3, 5, 6, 6, 7, 8, 9])
    # An even window is widened, and a window longer than the data is shrunk.
    assert np.allclose(_runmed(x, 2, "keep"), _runmed(x, 3, "keep"))
    assert np.allclose(_runmed(x, 99, "constant"), np.full(9, 6.0))


def test_runmed_endrules_differ_only_at_the_ends():
    rng = np.random.default_rng(0)
    x = rng.normal(100, 20, 500)
    a, b = _runmed(x, 51, "median"), _runmed(x, 51, "constant")
    assert np.allclose(a[25:-25], b[25:-25])
    assert not np.allclose(a[:25], b[:25])


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------
def test_normalise_beads_flattens_the_drift(cytof):
    """The synthetic file loses 45% sensitivity over the run; take it back out."""
    cytopy.normalise_beads(cytof)
    bead = np.asarray(cytof.obs["bead"], dtype=bool)
    time = np.asarray(cytof.X[:, cytopy.channel_index(cytof, "Time")])[bead]
    early, late = time < np.quantile(time, 0.25), time > np.quantile(time, 0.75)

    j = cytopy.channel_index(cytof, "Ce140Di")
    raw = np.asarray(cytof.X)[bead, j]
    assert np.median(raw[late]) / np.median(raw[early]) < 0.75  # drift is really there

    normed = np.asarray(cytof.layers["normalised"])[bead, j]
    assert np.median(normed[late]) / np.median(normed[early]) == pytest.approx(1.0, abs=0.05)


def test_normalise_beads_leaves_time_alone(cytof):
    cytopy.normalise_beads(cytof)
    for channel in ("Time", "Event_length"):
        j = cytopy.channel_index(cytof, channel)
        assert np.array_equal(cytof.layers["normalised"][:, j], cytof.X[:, j])
    j = cytopy.channel_index(cytof, "CD4")
    assert not np.allclose(cytof.layers["normalised"][:, j], cytof.X[:, j])


def test_normalise_beads_records_what_it_did(cytof):
    cytopy.normalise_beads(cytof)
    info = cytof.uns["cytopy"]["beads"]
    assert info["normalised_layer"] == "normalised"
    assert info["statistic"] == "median" and info["k"] == 201 and info["endrule"] == "median"
    assert set(info["baseline"]) == set(BEADS)
    assert "Time" not in info["scaled_channels"]
    assert cytof.obs["bead_slope"].between(0.5, 2.0).all()


def test_baseline_can_come_from_another_acquisition(cytof, cytof_path):
    other = cytopy.read_fcs(cytof_path)
    cytopy.gate_beads(other)
    baseline = cytopy.bead_baseline(other)
    baseline *= 2.0

    cytopy.normalise_beads(cytof, baseline=baseline)
    bead = np.asarray(cytof.obs["bead"], dtype=bool)
    j = cytopy.channel_index(cytof, "Ce140Di")
    assert np.median(cytof.layers["normalised"][bead, j]) == pytest.approx(
        2 * baseline["Ce140Di"] / 2, rel=0.05
    )
    assert cytof.uns["cytopy"]["beads"]["statistic"] == "given"


def test_baseline_statistic_and_bad_input(cytof):
    cytopy.gate_beads(cytof)
    assert not np.allclose(
        cytopy.bead_baseline(cytof, statistic="median").to_numpy(),
        cytopy.bead_baseline(cytof, statistic="mean").to_numpy(),
    )
    with pytest.raises(ValueError, match="median.*mean"):
        cytopy.bead_baseline(cytof, statistic="mode")
    with pytest.raises(KeyError, match="gate_beads"):
        cytopy.bead_baseline(cytof, bead_key="nope")
    with pytest.raises(ValueError, match="must be positive"):
        cytopy.normalise_beads(cytof, baseline=dict.fromkeys(BEADS, 0.0))
    with pytest.raises(KeyError, match="no value for bead channel"):
        cytopy.normalise_beads(cytof, baseline={"Ce140Di": 1.0})


def test_each_sample_gets_its_own_clock(cytof, cytof_path):
    """Time restarts with every file, so slopes must be interpolated per sample."""
    second = cytopy.read_fcs(cytof_path, sample_id="run2")
    second.X = second.X * np.float32(0.5)
    both = cytopy.concat_samples([cytof, second])

    cytopy.normalise_beads(both)
    per_sample = both.uns["cytopy"]["beads"]["samples"]
    assert set(per_sample) == {"demo_cytof", "run2"}
    # The dimmer run needs roughly twice the correction of the brighter one.
    ratio = per_sample["run2"]["median_slope"] / per_sample["demo_cytof"]["median_slope"]
    assert ratio == pytest.approx(2.0, rel=0.1)

    bead = np.asarray(both.obs["bead"], dtype=bool)
    sample = both.obs["sample"].astype(str).to_numpy()
    j = cytopy.channel_index(both, "Ce140Di")
    normed = np.asarray(both.layers["normalised"])[:, j]
    a = np.median(normed[bead & (sample == "demo_cytof")])
    b = np.median(normed[bead & (sample == "run2")])
    assert a == pytest.approx(b, rel=0.05)  # both land on the shared baseline


def test_a_sample_without_beads_is_an_error(cytof, cytof_path):
    """One bad file must not be normalised off another file's beads."""
    second = cytopy.read_fcs(cytof_path, sample_id="run2")
    both = cytopy.concat_samples([cytof, second])
    both.obs["bead"] = cytopy.gate_beads(both, key_added=None)
    both.obs.loc[both.obs["sample"].astype(str) == "run2", "bead"] = False
    with pytest.raises(ValueError, match="sample 'run2' has no bead events"):
        cytopy.normalise_beads(both)


def test_normalise_beads_needs_a_time_channel(cytof):
    trimmed = cytof[:, [c for c in cytof.var_names if "Time" not in str(c)]].copy()
    trimmed.var["kind"] = "fluor"
    with pytest.raises(KeyError, match="no Time channel"):
        cytopy.normalise_beads(trimmed)


# --------------------------------------------------------------------------
# distance and diagnostics
# --------------------------------------------------------------------------
def test_bead_distance_is_smallest_at_the_beads(cytof):
    cytopy.gate_beads(cytof)
    distance = cytopy.bead_distance(cytof)
    bead = np.asarray(cytof.obs["bead"], dtype=bool)
    assert np.median(distance[bead]) < np.median(distance[~bead]) / 5
    assert np.array_equal(np.asarray(cytof.obs["bead_distance"]), distance)
    with pytest.raises(KeyError, match="gate_beads"):
        cytopy.bead_distance(cytof, bead_key="nope")


def test_plot_beads_over_time_draws_before_and_after(cytof):
    """CATALYST's figure: a line per bead channel, with its baseline dashed."""
    cytopy.normalise_beads(cytof)
    fig = cytopy.plot_beads_over_time(cytof)
    assert [ax.get_title(loc="left") for ax in fig.axes] == ["before", "after"]
    solid = [ln for ln in fig.axes[0].lines if ln.get_linestyle() == "-"]
    dashed = [ln for ln in fig.axes[0].lines if ln.get_linestyle() == "--"]
    assert len(solid) == len(dashed) == len(BEADS)

    def drift(axis):
        """Start-to-end change of each curve, as a fraction of its baseline."""
        curves = [ln for ln in axis.lines if ln.get_linestyle() == "-"]
        bases = [ln.get_ydata()[0] for ln in axis.lines if ln.get_linestyle() == "--"]
        out = []
        for line, base in zip(curves, bases):
            values = np.asarray(line.get_ydata())
            edge = max(len(values) // 10, 1)
            out.append(abs(values[-edge:].mean() - values[:edge].mean()) / base)
        return np.array(out)

    # The before panel slopes away; the after panel sits on its baselines.
    assert (drift(fig.axes[0]) > 0.25).all()
    assert (drift(fig.axes[1]) < 0.05).all()


# --------------------------------------------------------------------------
# fidelity to the reference implementations
# --------------------------------------------------------------------------
# Produced by running premessa's identify_beads / smooth_beads /
# compute_bead_slopes / correct_data_channels and CATALYST's normCytof
# arithmetic, verbatim, on this same synthetic file. At float64 the Python
# here agrees with both to ~1e-12; the layer is stored as float32, so the
# assertions below allow for that and no more.
R_N_BEADS = 2480
R_BASELINE_MEDIAN = [168.8183593750, 139.0186080933, 199.0907669067, 115.1953620911, 155.0085372925]
R_BASELINE_MEAN = [170.5384097376, 140.0142273349, 201.4321270666, 116.3982931691, 155.3548427397]
R_EVENTS = [0, 10_000, 20_000, 30_000, 39_999]
R_PREMESSA_SLOPE = [0.7573452336, 0.8669049272, 0.9929214818, 1.1576189469, 1.3552456028]
R_CATALYST_SLOPE = [0.8141346967, 0.8720655562, 1.0056589758, 1.1674858885, 1.3021042790]
R_PREMESSA_CD4 = [0.717863, 0.844718, 2.915612, 0.065398, 0.406502]
R_CATALYST_CD4 = [0.771692, 0.849746, 2.953014, 0.065956, 0.390562]


def test_matches_the_r_reference_implementations(cytof):
    """The gate, the baseline, the slopes and the output, against R."""
    mask = cytopy.gate_beads(cytof)
    assert int(mask.sum()) == R_N_BEADS

    assert cytopy.bead_baseline(cytof).to_numpy() == pytest.approx(R_BASELINE_MEDIAN, rel=1e-9)
    assert cytopy.bead_baseline(cytof, statistic="mean").to_numpy() == pytest.approx(
        R_BASELINE_MEAN, rel=1e-9
    )

    j = cytopy.channel_index(cytof, "CD4")
    for kwargs, slopes, cd4 in [
        ({}, R_PREMESSA_SLOPE, R_PREMESSA_CD4),
        ({"statistic": "mean", "k": 501, "endrule": "constant"}, R_CATALYST_SLOPE, R_CATALYST_CD4),
    ]:
        cytopy.normalise_beads(cytof, key_added="check", **kwargs)
        assert cytof.obs["bead_slope"].to_numpy()[R_EVENTS] == pytest.approx(slopes, rel=1e-8)
        assert cytof.layers["check"][R_EVENTS, j] == pytest.approx(cd4, rel=1e-5)


def test_the_two_references_differ_where_documented(cytof):
    """premessa and CATALYST disagree on baseline, window and end rule -- only."""
    cytopy.normalise_beads(cytof, key_added="premessa")
    premessa = cytof.obs["bead_slope"].to_numpy().copy()
    cytopy.normalise_beads(cytof, statistic="mean", k=501, endrule="constant", key_added="catalyst")
    catalyst = cytof.obs["bead_slope"].to_numpy()
    assert not np.allclose(premessa, catalyst)
    assert np.median(np.abs(premessa - catalyst)) < 0.05  # same correction, different smoothing


def test_cli_normalises(cytof_path, monkeypatch, capsys):
    from cytopy import __main__

    seen = {}
    monkeypatch.setattr("cytopy.view", lambda adata, **kw: seen.update(adata=adata, **kw))

    assert __main__.main([str(cytof_path), "--normalise"]) == 0
    assert seen["layer"] == "normalised"
    assert "normalised" in seen["adata"].layers
    assert "beads" in capsys.readouterr().out

    assert __main__.main([str(cytof_path), "--normalise", "dvs", "--asinh"]) == 0
    assert seen["layer"] == "asinh"
    assert seen["adata"].uns["cytopy"]["beads"]["normalised_layer"] == "normalised"


# --------------------------------------------------------------------------
# behaviour that exists for large runs
# --------------------------------------------------------------------------
def test_sample_groups_without_stringifying_every_event(cytof):
    """Grouping goes via categorical codes, so it must handle the awkward cases."""
    from cytopy.beads import _sample_groups

    n = cytof.n_obs
    assert _sample_groups(cytof, None) == [("", pytest.approx(np.arange(n)))]
    assert _sample_groups(cytof, "absent")[0][0] == ""

    labels = np.where(np.arange(n) % 3 == 0, "a", "b")
    cytof.obs["three"] = pd.Categorical(labels, categories=["a", "b", "unused"])
    groups = dict(_sample_groups(cytof, "three"))
    assert sorted(groups) == ["a", "b"]  # an unused category contributes no group
    assert np.array_equal(groups["a"], np.flatnonzero(labels == "a"))
    assert np.array_equal(groups["b"], np.flatnonzero(labels == "b"))
    # rows stay in file order, which is what the time-ordered smoothing assumes
    assert all(np.all(np.diff(rows) > 0) for rows in groups.values())

    cytof.obs["plain"] = labels  # not a categorical
    assert sorted(dict(_sample_groups(cytof, "plain"))) == ["a", "b"]

    missing = pd.Categorical([None] + list(labels[1:]), categories=["a", "b"])
    cytof.obs["gappy"] = missing
    assert "<missing>" in dict(_sample_groups(cytof, "gappy"))


def test_normalise_in_place_avoids_the_extra_layer(cytof):
    """key_added=None overwrites X, which halves peak memory on a big run."""
    before = np.asarray(cytof.X).copy()
    cytopy.normalise_beads(cytof, key_added=None)
    assert "normalised" not in cytof.layers
    j = cytopy.channel_index(cytof, "Ce140Di")
    assert not np.allclose(cytof.X[:, j], before[:, j])
    slopes = cytof.obs["bead_slope"].to_numpy()
    assert cytof.X[:, j] == pytest.approx(before[:, j] * slopes, rel=1e-6)
    assert np.array_equal(cytof.X[:, 0], before[:, 0])  # Time still untouched


def test_normalise_does_not_upcast_the_whole_matrix(cytof):
    """The output is written at the storage dtype, not float64."""
    assert cytof.X.dtype == np.float32
    cytopy.normalise_beads(cytof)
    assert cytof.layers["normalised"].dtype == np.float32


def test_plot_thins_the_drawing_but_not_the_smoothing(cytof):
    cytopy.normalise_beads(cytof)
    n_beads = int(cytof.obs["bead"].sum())

    full = cytopy.plot_beads_over_time(cytof, max_points=0)
    assert len(full.axes[0].lines[0].get_xdata()) == n_beads

    thin = cytopy.plot_beads_over_time(cytof, max_points=500)
    x, y = thin.axes[0].lines[0].get_xdata(), thin.axes[0].lines[0].get_ydata()
    assert len(x) <= 500
    # The thinned curve is a subset of the full one: the running median still
    # sees every bead event, only the drawing skips points.
    step = int(np.ceil(n_beads / 500))
    assert np.allclose(x, full.axes[0].lines[0].get_xdata()[::step])
    assert np.allclose(y, full.axes[0].lines[0].get_ydata()[::step])
