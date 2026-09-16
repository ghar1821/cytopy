"""Getting a spillover matrix: from controls, from the file, from someone else's CSV."""

import numpy as np
import pandas as pd
import pytest

import cytopy

FLUOR = ["CD3 (FITC-A)", "CD19 (PE-A)", "CD8 (APC-A)"]


def _matrix(values, names=FLUOR):
    return pd.DataFrame(np.asarray(values, dtype=float), index=names, columns=names)


# --------------------------------------------------------------------------
# deriving from single-stain controls
# --------------------------------------------------------------------------
def test_read_controls_matches_each_file_to_its_detector(controls):
    stained, unstained = controls
    assert sorted(stained) == sorted(FLUOR)
    assert unstained is not None
    assert (unstained.obs["sample"] == "Unstained").all()
    assert stained["CD3 (FITC-A)"].obs["sample"].iloc[0] == "Compensation Controls_FITC-A"
    # The controls carry no matrix of their own; that is the point of them.
    assert "spillover" not in stained["CD3 (FITC-A)"].uns


def test_read_controls_falls_back_to_the_signal_when_the_name_says_nothing(controls_dir, tmp_path):
    """A file named after nothing in the panel is matched by which channel splits."""
    import shutil

    for path in controls_dir.glob("*.fcs"):
        name = "Tube_002.fcs" if "PE-A" in path.name else path.name
        shutil.copy(path, tmp_path / name)
    stained, unstained = cytopy.read_controls(tmp_path)
    assert stained["CD19 (PE-A)"].obs["sample"].iloc[0] == "Tube_002"
    assert unstained is not None


def test_spillover_from_controls_recovers_the_true_matrix(controls, true_spillover):
    stained, _ = controls
    spill = cytopy.spillover_from_controls(stained)
    assert list(spill.index) == FLUOR and list(spill.columns) == FLUOR
    assert np.allclose(np.diag(spill), 1.0)
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.01)


def test_spillover_from_controls_against_an_unstained_reference(controls, true_spillover):
    stained, unstained = controls
    spill = cytopy.spillover_from_controls(stained, unstained=unstained)
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.01)
    assert spill.attrs["cytopy"]["negative"] == "unstained"


def test_spillover_from_controls_with_means(controls, true_spillover):
    stained, unstained = controls
    spill = cytopy.spillover_from_controls(stained, unstained=unstained, statistic="mean")
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.005)
    with pytest.raises(ValueError, match="median.*mean"):
        cytopy.spillover_from_controls(stained, statistic="mode")


def test_spillover_from_controls_accepts_markers_and_paths(controls, controls_dir, true_spillover):
    stained, _ = controls
    by_marker = {"CD3": stained[FLUOR[0]], "CD19": stained[FLUOR[1]], "APC-A": stained[FLUOR[2]]}
    assert np.allclose(
        cytopy.spillover_from_controls(by_marker).to_numpy(), true_spillover, atol=0.01
    )
    by_path = {"FITC-A": controls_dir / "Compensation Controls_FITC-A.fcs"}
    assert cytopy.spillover_from_controls(by_path).shape == (1, 1)


def test_spillover_from_controls_uses_a_gate_when_given_one(controls, true_spillover):
    stained, _ = controls
    for name, control in stained.items():
        j = cytopy.channel_index(control, name)
        control.obs["P1"] = np.asarray(control.X[:, j]) > 1000.0
    spill = cytopy.spillover_from_controls(stained, positive_gate="P1")
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.01)
    counts = spill.attrs["cytopy"]["controls"][FLUOR[0]]
    assert counts["positive_events"] == int((stained[FLUOR[0]].obs["P1"]).sum())


def test_spillover_from_controls_takes_explicit_thresholds(controls, true_spillover):
    stained, _ = controls
    spill = cytopy.spillover_from_controls(stained, thresholds=dict.fromkeys(stained, 1000.0))
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.01)


def test_an_all_positive_bead_control_needs_an_unstained_tube(controls, true_spillover):
    """A tube with no negatives of its own: -inf takes every event as positive."""
    stained, unstained = controls
    beads = {}
    for name, control in stained.items():
        column = np.asarray(control.X[:, cytopy.channel_index(control, name)])
        beads[name] = control[column > 1000].copy()

    spill = cytopy.spillover_from_controls(
        beads, unstained=unstained, thresholds=dict.fromkeys(beads, -np.inf)
    )
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.01)
    with pytest.raises(ValueError, match="negative events"):
        cytopy.spillover_from_controls(beads, thresholds=dict.fromkeys(beads, -np.inf))


def test_a_detector_without_a_control_only_appears_if_asked_for(controls):
    stained, _ = controls
    partial = {k: v for k, v in stained.items() if k != FLUOR[1]}
    assert list(cytopy.spillover_from_controls(partial).columns) == [FLUOR[0], FLUOR[2]]

    spill = cytopy.spillover_from_controls(partial, channels=FLUOR)
    assert list(spill.columns) == FLUOR
    # The unconstrained detector gets an identity row: it spills into nothing.
    assert np.allclose(spill.loc[FLUOR[1]], [0.0, 1.0, 0.0])
    # ... but the spill of the other dyes *into* it is still measured.
    assert spill.loc[FLUOR[0], FLUOR[1]] == pytest.approx(0.12, abs=0.01)


def test_spillover_from_controls_rejects_nonsense(controls):
    stained, unstained = controls
    with pytest.raises(ValueError, match="no controls"):
        cytopy.spillover_from_controls({})
    with pytest.raises(ValueError, match="does not separate into two populations"):
        cytopy.spillover_from_controls({"CD3": unstained})
    # A gate that picks the dim events instead of the bright ones.
    control = stained[FLUOR[0]].copy()
    column = np.asarray(control.X[:, cytopy.channel_index(control, FLUOR[0])])
    control.obs["inverted"] = column < np.median(column)
    with pytest.raises(ValueError, match="not brighter than its negatives"):
        cytopy.spillover_from_controls({FLUOR[0]: control}, positive_gate="inverted")
    with pytest.raises(ValueError, match="omits detectors"):
        cytopy.spillover_from_controls(stained, channels=[FLUOR[0]])
    with pytest.raises(ValueError, match="same detector"):
        cytopy.spillover_from_controls({"CD3": stained[FLUOR[0]], "FITC-A": stained[FLUOR[0]]})
    with pytest.raises(ValueError, match="positive events"):
        cytopy.spillover_from_controls(stained, min_events=10**9)


def test_a_derived_matrix_compensates_like_the_files_own(demo, controls):
    stained, unstained = controls
    spill = cytopy.spillover_from_controls(stained, unstained=unstained, statistic="mean")
    cytopy.compensate(demo, spill, key_added="derived")
    cytopy.compensate(demo, key_added="from_file")
    j = [cytopy.channel_index(demo, c) for c in FLUOR]
    scale = np.ptp(demo.layers["from_file"][:, j])
    assert (
        np.abs(demo.layers["derived"][:, j] - demo.layers["from_file"][:, j]).max() < 0.005 * scale
    )


# --------------------------------------------------------------------------
# reading someone else's matrix
# --------------------------------------------------------------------------
def test_write_then_read_spillover_roundtrips(demo, tmp_path):
    path = cytopy.write_spillover(demo.uns["spillover"], tmp_path / "spill.csv")
    back = cytopy.read_spillover(path)
    assert list(back.columns) == ["FITC-A", "PE-A", "APC-A"]
    assert np.allclose(back.to_numpy(), demo.uns["spillover"].to_numpy())


@pytest.mark.parametrize(
    ("name", "text"),
    [
        # FlowJo: percentages, "Comp-" prefixed headers, no row-name column
        ("flowjo.csv", "Comp-FITC-A,Comp-PE-A,Comp-APC-A\n100,12,0\n5,100,8\n0,0,100\n"),
        # Tab separated, detector :: marker headers, row names
        (
            "diva.tsv",
            (
                "\tFITC-A :: CD3\tPE-A :: CD19\tAPC-A :: CD8\n"
                "FITC-A :: CD3\t1\t0.12\t0\n"
                "PE-A :: CD19\t0.05\t1\t0.08\n"
                "APC-A :: CD8\t0\t0\t1\n"
            ),
        ),
        # Semicolon separated, as a European locale export writes it
        (
            "euro.csv",
            "detector;FITC-A;PE-A;APC-A\nFITC-A;1;0.12;0\nPE-A;0.05;1;0.08\nAPC-A;0;0;1\n",
        ),
        # A bare $SPILLOVER payload dumped to a file
        ("keyword.csv", "3,FITC-A,PE-A,APC-A,1,0.12,0,0.05,1,0.08,0,0,1\n"),
        # Comments and blank lines
        (
            "commented.csv",
            (
                "# exported by something\n\n"
                "detector,FITC-A,PE-A,APC-A\n"
                "FITC-A,1,0.12,0\nPE-A,0.05,1,0.08\nAPC-A,0,0,1\n"
            ),
        ),
    ],
)
def test_read_spillover_handles_what_other_software_writes(tmp_path, true_spillover, name, text):
    (tmp_path / name).write_text(text)
    spill = cytopy.read_spillover(tmp_path / name)
    assert list(spill.index) == list(spill.columns) == ["FITC-A", "PE-A", "APC-A"]
    assert np.allclose(spill.to_numpy(), true_spillover, atol=1e-9)


def test_read_spillover_can_invert_a_compensation_matrix(tmp_path, true_spillover):
    names = ["FITC-A", "PE-A", "APC-A"]
    comp = pd.DataFrame(np.linalg.inv(true_spillover), index=names, columns=names)
    cytopy.write_spillover(comp, tmp_path / "comp.csv")
    spill = cytopy.read_spillover(tmp_path / "comp.csv", inverted=True)
    assert np.allclose(spill.to_numpy(), true_spillover, atol=1e-9)


def test_read_spillover_percentages_can_be_forced(tmp_path):
    (tmp_path / "m.csv").write_text("d,A,B\nA,1,0.12\nB,0.05,1\n")
    assert cytopy.read_spillover(tmp_path / "m.csv", percent=True).loc["A", "A"] == 0.01


def test_read_spillover_rejects_bad_files(tmp_path):
    (tmp_path / "empty.csv").write_text("\n\n")
    with pytest.raises(ValueError, match="empty"):
        cytopy.read_spillover(tmp_path / "empty.csv")
    (tmp_path / "oblong.csv").write_text("d,A,B,C\nA,1,0,0\nB,0,1,0\n")
    with pytest.raises(ValueError, match="not square"):
        cytopy.read_spillover(tmp_path / "oblong.csv")
    (tmp_path / "zero.csv").write_text("d,A,B\nA,0,0.12\nB,0.05,1\n")
    with pytest.raises(ValueError, match="zero on the diagonal"):
        cytopy.read_spillover(tmp_path / "zero.csv")
    (tmp_path / "one.csv").write_text("not a spillover keyword\n")
    with pytest.raises(ValueError, match="SPILLOVER payload"):
        cytopy.read_spillover(tmp_path / "one.csv")


# --------------------------------------------------------------------------
# applying it
# --------------------------------------------------------------------------
def test_compensate_reads_a_csv_path(demo, tmp_path):
    path = cytopy.write_spillover(demo.uns["spillover"], tmp_path / "spill.csv")
    cytopy.compensate(demo, key_added="from_file")
    cytopy.compensate(demo, path, key_added="from_csv")
    assert np.allclose(demo.layers["from_csv"], demo.layers["from_file"])
    assert demo.uns["cytopy"]["spillover_source"] == f"csv:{path}"


def test_compensate_records_where_the_matrix_came_from(demo):
    cytopy.compensate(demo)
    assert demo.uns["cytopy"]["spillover_source"] == "uns"
    cytopy.compensate(demo, demo.uns["spillover"])
    assert demo.uns["cytopy"]["spillover_source"] == "argument"


def test_compensate_aligns_rows_to_columns(demo):
    """Rows in a different order than the columns must not silently transpose the fix."""
    spill = _matrix([[1.0, 0.2, 0.0], [0.05, 1.0, 0.1], [0.0, 0.03, 1.0]])
    cytopy.compensate(demo, spill, key_added="ordered")
    cytopy.compensate(demo, spill.iloc[[2, 0, 1]], key_added="scrambled")
    assert np.allclose(demo.layers["ordered"], demo.layers["scrambled"])


def test_compensate_rejects_a_matrix_it_cannot_use(demo):
    with pytest.raises(ValueError, match="square"):
        cytopy.compensate(demo, pd.DataFrame(np.ones((2, 3))))
    with pytest.raises(ValueError, match="do not name the same detectors"):
        cytopy.compensate(demo, _matrix(np.eye(3)).rename(columns={FLUOR[0]: "FSC-A"}))
    with pytest.raises(ValueError, match="singular"):
        cytopy.compensate(demo, _matrix(np.ones((3, 3))))
    with pytest.raises(ValueError, match="fluorescence channels"):
        cytopy.compensate(demo, np.eye(2))
    demo.uns.pop("spillover")
    with pytest.raises(ValueError, match="no spillover matrix given"):
        cytopy.compensate(demo)


def test_cli_derives_and_applies_a_matrix(demo_path, controls_dir, tmp_path, monkeypatch, capsys):
    """--controls and --spillover both end up compensating; they are exclusive."""
    from cytopy import __main__

    seen = {}
    monkeypatch.setattr("cytopy.view", lambda adata, **kw: seen.update(adata=adata, **kw))

    assert __main__.main([str(demo_path), "--controls", str(controls_dir)]) == 0
    assert seen["layer"] == "comp"
    assert seen["adata"].uns["cytopy"]["spillover_source"] == "argument"
    assert "CD3 (FITC-A)" in capsys.readouterr().out

    path = cytopy.write_spillover(cytopy.read_fcs(demo_path).uns["spillover"], tmp_path / "s.csv")
    assert __main__.main([str(demo_path), "--spillover", str(path)]) == 0
    assert seen["adata"].uns["cytopy"]["spillover_source"] == f"csv:{path}"

    with pytest.raises(SystemExit):
        __main__.main([str(demo_path), "--compensate", "--controls", str(controls_dir)])
