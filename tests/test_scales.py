import numpy as np
import pytest

from cytopy.scales import AsinhScale, LinearScale, LogicleScale, LogScale, get_scale


@pytest.mark.parametrize(
    "scale",
    [LinearScale(), LogScale(), AsinhScale(cofactor=150.0), LogicleScale(T=262144, W=0.5, M=4.5)],
)
def test_roundtrip(scale):
    data = np.array([-1000.0, -10.0, 0.0, 1.0, 100.0, 5000.0, 200000.0])
    if isinstance(scale, LogScale):
        data = data[data > 0]
    out = scale.inverse(scale.forward(data))
    assert np.allclose(out, data, rtol=1e-3, atol=1e-2)


def test_logicle_anchor_points():
    s = LogicleScale(T=262144.0, W=0.5, M=4.5, A=0.0)
    assert s.inverse(np.array(s.x1)) == pytest.approx(0.0, abs=1e-9)
    assert s.inverse(np.array(1.0)) == pytest.approx(262144.0, rel=1e-9)
    assert s.forward(np.array(0.0)) == pytest.approx(s.x1, abs=1e-4)


def test_logicle_is_monotonic():
    s = LogicleScale(T=1e5, W=1.0, M=4.5)
    x = np.linspace(-1e4, 1e5, 20001)
    assert np.all(np.diff(s.forward(x)) >= 0)


def test_logicle_from_data_widens_for_negatives():
    # With the top of scale fixed, a broader negative population must buy more
    # decades of linearisation around zero.
    rng = np.random.default_rng(0)
    tight = rng.normal(0, 10, 10000)
    wide = rng.normal(0, 2000, 10000)
    assert LogicleScale.from_data(wide, T=1e5).W > LogicleScale.from_data(tight, T=1e5).W


def test_logicle_from_data_handles_all_positive_data():
    s = LogicleScale.from_data(np.abs(np.random.default_rng(1).normal(1e3, 10, 5000)))
    assert 0 <= s.W <= s.M / 2


def test_ticks_are_inside_limits_and_labelled():
    s = LogicleScale(T=262144.0, W=0.5, M=4.5)
    lo, hi = s.forward(np.array([-1000.0, 262144.0]))
    t = s.ticks(float(lo), float(hi))
    assert len(t.major) == len(t.labels)
    assert np.all((t.major >= lo) & (t.major <= hi))
    assert "0" in t.labels and any("10" in lbl for lbl in t.labels)


def test_get_scale_dispatch():
    assert isinstance(get_scale("biex", np.array([-5.0, 1e4])), LogicleScale)
    assert get_scale("asinh", cofactor=5).cofactor == 5
    with pytest.raises(ValueError):
        get_scale("nope")
