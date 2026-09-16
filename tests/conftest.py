import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))


@pytest.fixture(scope="session")
def demo_path(tmp_path_factory):
    from make_demo_fcs import main

    path = tmp_path_factory.mktemp("data") / "demo.fcs"
    main(str(path))
    return path


@pytest.fixture
def demo(demo_path):
    import cytopy

    return cytopy.read_fcs(demo_path)


@pytest.fixture(scope="session")
def controls_dir(tmp_path_factory):
    """A directory of single-stain controls plus an unstained one, no $SPILLOVER."""
    from make_demo_fcs import write_controls

    path = tmp_path_factory.mktemp("controls")
    write_controls(path)
    return path


@pytest.fixture(scope="session")
def true_spillover():
    from make_demo_fcs import spillover_matrix

    return spillover_matrix()


@pytest.fixture
def controls(controls_dir):
    import cytopy

    return cytopy.read_controls(controls_dir)


@pytest.fixture(scope="session")
def cytof_path(tmp_path_factory):
    """A synthetic mass cytometry file: EQ beads, DNA, and sensitivity drift."""
    from make_demo_cytof import main

    path = tmp_path_factory.mktemp("cytof") / "demo_cytof.fcs"
    main(str(path))
    return path


@pytest.fixture
def cytof(cytof_path):
    import cytopy

    return cytopy.read_fcs(cytof_path)


@pytest.fixture(scope="session")
def barcoded_path(tmp_path_factory):
    """Synthetic CyTOF with a 3-of-6 palladium barcode, beads, drift and doublets."""
    from make_demo_cytof import write_barcoded

    path = tmp_path_factory.mktemp("barcoded") / "demo_barcoded.fcs"
    write_barcoded(str(path))
    return path


@pytest.fixture(scope="session")
def barcoded_truth(barcoded_path, tmp_path_factory):
    """The barcode each event really carries; "0" for beads and doublets."""
    import sys

    from make_demo_cytof import make_barcoded_events

    del sys
    return make_barcoded_events(60_000)[3]


@pytest.fixture
def barcoded(barcoded_path):
    import cytopy

    return cytopy.read_fcs(barcoded_path)


@pytest.fixture(autouse=True)
def _close_figures():
    """Plot functions hand back open figures; the caller closes them, so do we."""
    yield
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    plt.close("all")
