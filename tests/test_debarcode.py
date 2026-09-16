"""Debarcoding, checked against CATALYST.

``test_matches_the_r_reference`` pins values produced by running CATALYST's
``assignPrelim`` / ``estCutoffs`` / ``applyCutoffs`` -- the verbatim R source,
``drc`` included -- on this same synthetic file.
"""

import numpy as np
import pandas as pd
import pytest

import cytopy

BARCODE_CHANNELS = ["Pd102Di", "Pd104Di", "Pd105Di", "Pd106Di", "Pd108Di", "Pd110Di"]


# --------------------------------------------------------------------------
# the scheme
# --------------------------------------------------------------------------
def test_pd20_is_every_three_of_six():
    key = cytopy.barcode_key("pd20")
    assert key.shape == (20, 6)
    assert list(key.columns) == ["102", "104", "105", "106", "108", "110"]
    assert (key.sum(axis=1) == 3).all()
    assert key.index[0] == "A1" and key.index[-1] == "A20"
    assert len(set(map(tuple, key.to_numpy()))) == 20  # every barcode distinct


def test_a_key_can_be_given_as_a_mapping_or_a_table():
    mapping = cytopy.barcode_key({"x": [102, 104], "y": [104, 105]})
    assert list(mapping.index) == ["x", "y"]
    assert mapping.loc["x"].tolist() == [1, 1, 0]

    frame = pd.DataFrame([[1, 0], [0, 1]], index=["a", "b"], columns=[102, 104])
    assert cytopy.barcode_key(frame).loc["b"].tolist() == [0, 1]
    assert list(cytopy.barcode_key(frame, ids=["p", "q"]).index) == ["p", "q"]


def test_bad_schemes_are_rejected():
    with pytest.raises(KeyError, match="unknown barcoding scheme"):
        cytopy.barcode_key("nonesuch")
    with pytest.raises(ValueError, match="only 0 and 1"):
        cytopy.barcode_key(pd.DataFrame([[2, 0]], index=["a"], columns=[102, 104]))
    with pytest.raises(ValueError, match="reserved for unassigned"):
        cytopy.barcode_key(pd.DataFrame([[1, 0]], index=["0"], columns=[102, 104]))
    with pytest.raises(ValueError, match="20 barcodes but 2 ids"):
        cytopy.barcode_key("pd20", ids=["a", "b"])


def test_a_missing_barcode_channel_is_fatal(cytof):
    with pytest.raises(KeyError, match="no channel for barcode mass"):
        cytopy.assign_prelim(cytof, "pd20")


# --------------------------------------------------------------------------
# assignment
# --------------------------------------------------------------------------
def test_assign_prelim_recovers_the_true_barcodes(barcoded, barcoded_truth):
    cytopy.assign_prelim(barcoded, "pd20")
    assigned = np.asarray(barcoded.obs["bc_id"], dtype=object)
    cells = barcoded_truth != "0"
    # Every real cell gets its own barcode back; beads and doublets are the
    # ones the cutoffs still have to deal with.
    assert (assigned[cells] == barcoded_truth[cells]).mean() > 0.99
    assert np.isfinite(barcoded.obs["delta"]).all()
    assert barcoded.obs["delta"].between(0, 1.5).all()


def test_assign_prelim_records_the_scheme(barcoded):
    cytopy.assign_prelim(barcoded, "pd20")
    info = barcoded.uns["cytopy"]["debarcode"]
    assert info["channels"] == BARCODE_CHANNELS
    assert info["n_positive"] == 3
    assert info["cofactor"] == 5.0
    assert sum(info["key"]["A1"]) == 3
    # The preliminary call is kept so yield plots stay honest after cutoffs.
    assert np.array_equal(
        np.asarray(barcoded.obs["bc_prelim"], dtype=object),
        np.asarray(barcoded.obs["bc_id"], dtype=object),
    )


def test_the_scaled_layer_is_optional(barcoded):
    cytopy.assign_prelim(barcoded, "pd20")
    assert "scaled" not in barcoded.layers  # a whole extra matrix, off by default
    cytopy.assign_prelim(barcoded, "pd20", scaled_layer="scaled")
    assert barcoded.layers["scaled"].shape == barcoded.shape


# --------------------------------------------------------------------------
# cutoffs
# --------------------------------------------------------------------------
def test_estimate_cutoffs_lands_between_the_doublets_and_the_cells(barcoded):
    cytopy.assign_prelim(barcoded, "pd20")
    cutoffs = cytopy.estimate_cutoffs(barcoded)
    assert len(cutoffs) == 20
    assert cutoffs.between(0.3, 0.8).all()
    assert np.allclose(
        cutoffs.to_numpy(),
        pd.Series(barcoded.uns["cytopy"]["debarcode"]["sep_cutoffs"]).to_numpy(),
    )


def test_apply_cutoffs_throws_out_the_doublets(barcoded, barcoded_truth):
    cytopy.debarcode(barcoded, "pd20")
    final = np.asarray(barcoded.obs["bc_id"], dtype=object)
    assigned = final != "0"
    # What survives is almost entirely correct...
    assert (final[assigned] == barcoded_truth[assigned]).mean() > 0.99
    # ...and the doublets are nearly all gone.
    doublets = barcoded_truth == "0"
    assert (assigned & doublets).sum() / doublets.sum() < 0.05
    assert np.isfinite(barcoded.obs["mhl_dist"][assigned]).all()


def test_a_stricter_cutoff_assigns_fewer_events(barcoded):
    cytopy.assign_prelim(barcoded, "pd20")
    cytopy.apply_cutoffs(barcoded, sep_cutoffs=0.2, mhl_cutoff=30)
    loose = int((np.asarray(barcoded.obs["bc_id"], dtype=object) != "0").sum())

    cytopy.assign_prelim(barcoded, "pd20")
    cytopy.apply_cutoffs(barcoded, sep_cutoffs=0.8, mhl_cutoff=30)
    tight = int((np.asarray(barcoded.obs["bc_id"], dtype=object) != "0").sum())
    assert tight < loose


def test_the_mahalanobis_cutoff_bites(barcoded):
    cytopy.assign_prelim(barcoded, "pd20")
    cytopy.apply_cutoffs(barcoded, sep_cutoffs=0.0, mhl_cutoff=1e9)
    everything = int((np.asarray(barcoded.obs["bc_id"], dtype=object) != "0").sum())

    cytopy.assign_prelim(barcoded, "pd20")
    cytopy.apply_cutoffs(barcoded, sep_cutoffs=0.0, mhl_cutoff=5)
    trimmed = int((np.asarray(barcoded.obs["bc_id"], dtype=object) != "0").sum())
    assert trimmed < everything
    assert barcoded.uns["cytopy"]["debarcode"]["mhl_cutoff"] == 5


def test_cutoffs_can_be_given_per_barcode(barcoded):
    cytopy.assign_prelim(barcoded, "pd20")
    cytopy.apply_cutoffs(barcoded, sep_cutoffs={"A1": 0.99, "A2": 0.0})
    used = barcoded.uns["cytopy"]["debarcode"]["sep_cutoffs_used"]
    assert used["A1"] == 0.99 and used["A2"] == 0.0
    assert used["A3"] == 1.0  # unmentioned barcodes are discarded, not guessed
    with pytest.raises(KeyError, match="unknown barcode"):
        cytopy.apply_cutoffs(barcoded, sep_cutoffs={"Z9": 0.5})


def test_debarcoding_records_what_it_removed(barcoded):
    cytopy.debarcode(barcoded, "pd20")
    log = cytopy.filter_log(barcoded)
    assert list(log["step"]) == ["debarcode"]
    row = log.iloc[0]
    assert row["n_before"] == barcoded.n_obs
    assert row["n_after"] == int((np.asarray(barcoded.obs["bc_id"], dtype=object) != "0").sum())
    assert "separation cutoff" in row["reason"] and "Mahalanobis" in row["reason"]


def test_steps_must_run_in_order(barcoded):
    with pytest.raises(KeyError, match="assign_prelim first"):
        cytopy.estimate_cutoffs(barcoded)
    with pytest.raises(KeyError, match="assign_prelim first"):
        cytopy.apply_cutoffs(barcoded)
    cytopy.assign_prelim(barcoded, "pd20")
    with pytest.raises(KeyError, match="no separation cutoffs"):
        cytopy.apply_cutoffs(barcoded)


def test_barcode_stats(barcoded):
    cytopy.debarcode(barcoded, "pd20")
    stats = cytopy.barcode_stats(barcoded)
    assert list(stats.index[:1]) == ["0"]
    assert stats["n"].sum() == barcoded.n_obs
    assert stats.loc["A1", "channels"] == "Pd102Di+Pd104Di+Pd105Di"
    assert 0 < stats.loc["A1", "sep_cutoff"] < 1


# --------------------------------------------------------------------------
# fidelity to CATALYST
# --------------------------------------------------------------------------
# From running CATALYST's assignPrelim / estCutoffs / applyCutoffs verbatim in
# R -- drc included for the LL.3 fit -- on this same synthetic file.
R_EVENTS = [0, 10_000, 25_000, 40_000, 59_999]
R_PRELIM = ["A9", "A18", "A12", "A7", "A20"]
R_FINAL = ["A9", "A18", "A12", "A7", "A20"]
R_DELTA = [0.7997275994, 0.6584063441, 0.6830413591, 0.8272781626, 0.759360129]
R_MHL = [2.25638115, 19.08692088, 10.53195244, 4.45439396, 4.65658365]
R_N_PRELIM = 60_000
R_N_FINAL = 52_753
R_CUTOFFS = {
    "A1": 0.530579,
    "A2": 0.530757,
    "A3": 0.540455,
    "A4": 0.540524,
    "A5": 0.530542,
    "A6": 0.540529,
    "A7": 0.540388,
    "A8": 0.530541,
    "A9": 0.540591,
    "A10": 0.530621,
    "A11": 0.530558,
    "A12": 0.530639,
    "A13": 0.530582,
    "A14": 0.530664,
    "A15": 0.530616,
    "A16": 0.530585,
    "A17": 0.540567,
    "A18": 0.530583,
    "A19": 0.530564,
    "A20": 0.540512,
}


def test_matches_the_r_reference(barcoded):
    """Assignments, deltas and distances against CATALYST's own code."""
    cytopy.assign_prelim(barcoded, "pd20")
    prelim = np.asarray(barcoded.obs["bc_id"], dtype=object)
    assert int((prelim != "0").sum()) == R_N_PRELIM
    assert list(prelim[R_EVENTS]) == R_PRELIM
    assert barcoded.obs["delta"].to_numpy()[R_EVENTS] == pytest.approx(R_DELTA, abs=1e-9)

    cytopy.apply_cutoffs(barcoded, sep_cutoffs=R_CUTOFFS, mhl_cutoff=30)
    final = np.asarray(barcoded.obs["bc_id"], dtype=object)
    assert int((final != "0").sum()) == R_N_FINAL
    assert list(final[R_EVENTS]) == R_FINAL
    assert barcoded.obs["mhl_dist"].to_numpy()[R_EVENTS] == pytest.approx(R_MHL, rel=1e-9)


def test_cutoff_estimates_track_the_r_reference(barcoded):
    """scipy's log-logistic fit is not drc's, so the last digits can differ.

    Nineteen of the twenty agree to 1e-5. One lands on the adjacent point of
    the 0.01 grid the estimate is searched on, which changes no assignment --
    :func:`test_matches_the_r_reference` runs the R cutoffs through and gets
    the same events either way.
    """
    cytopy.assign_prelim(barcoded, "pd20")
    mine = cytopy.estimate_cutoffs(barcoded)
    difference = (mine - pd.Series(R_CUTOFFS)).abs()
    assert (difference <= 0.01).all()
    assert int((difference > 1e-4).sum()) <= 1

    # Estimated or borrowed, the debarcoding comes out the same.
    cytopy.apply_cutoffs(barcoded, mhl_cutoff=30)
    assert int((np.asarray(barcoded.obs["bc_id"], dtype=object) != "0").sum()) == pytest.approx(
        R_N_FINAL, rel=0.01
    )


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def test_plot_yields(barcoded):
    cytopy.debarcode(barcoded, "pd20")
    fig = cytopy.plot_yields(barcoded, ["A1", "A2"])
    panels = [ax for ax in fig.axes if ax.get_title(loc="left")]
    assert len(panels) == 2
    assert "A1: Pd102Di+Pd104Di+Pd105Di" in panels[0].get_title(loc="left")
    # grey bars, a red yield curve, and the cutoff marked
    assert len(panels[0].patches) > 50
    assert any(line.get_color() == "red" for line in panels[0].lines)
    assert any(line.get_linestyle() == "--" for line in panels[0].lines)

    assert len(cytopy.plot_yields(barcoded, "all").axes) == 2  # one panel plus its twin
    with pytest.raises(KeyError, match="unknown barcode"):
        cytopy.plot_yields(barcoded, ["Z9"])


def test_yield_curves_are_drawn_against_the_preliminary_call(barcoded):
    """Otherwise every curve reads 100% once the cutoffs have been applied."""
    cytopy.assign_prelim(barcoded, "pd20")
    before = cytopy.plot_yields(barcoded, ["A1"]).axes[0]
    counts_before = sum(p.get_height() for p in before.patches)

    cytopy.estimate_cutoffs(barcoded)
    cytopy.apply_cutoffs(barcoded)
    after = cytopy.plot_yields(barcoded, ["A1"]).axes[0]
    assert sum(p.get_height() for p in after.patches) == counts_before


def test_plot_barcode_events(barcoded):
    cytopy.debarcode(barcoded, "pd20")
    fig = cytopy.plot_barcode_events(barcoded, ["0", "A1"], n=40)
    panels = [ax for ax in fig.axes if ax.get_title(loc="left")]
    assert "unassigned" in panels[0].get_title(loc="left")
    assert "A1" in panels[1].get_title(loc="left")
    assert len(panels[1].collections) == len(BARCODE_CHANNELS)
    assert all(c.get_offsets().shape[0] <= 40 for c in panels[1].collections)
    with pytest.raises(KeyError, match="unknown barcode"):
        cytopy.plot_barcode_events(barcoded, ["Z9"])
