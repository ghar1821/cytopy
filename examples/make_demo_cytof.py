"""Write a synthetic mass cytometry FCS file with EQ beads and signal drift.

Fluidigm-style channel names, a DNA intercalator, three markers and the five
Fluidigm bead channels. Every mass channel decays over the run, which is the
drift bead normalisation exists to take out.

    python examples/make_demo_cytof.py demo_cytof.fcs
"""

from __future__ import annotations

import sys
from pathlib import Path

import flowio
import numpy as np

CHANNELS = [
    "Time",
    "Event_length",
    "Ce140Di",
    "Eu151Di",
    "Eu153Di",
    "Ho165Di",
    "Lu175Di",
    "Nd142Di",
    "Sm147Di",
    "Er168Di",
    "Ir191Di",
    "Ir193Di",
]
MARKERS = ["", "", "", "", "", "", "", "CD19", "CD4", "CD8", "DNA1", "DNA2"]
BEAD_CHANNELS = ["Ce140Di", "Eu151Di", "Eu153Di", "Ho165Di", "Lu175Di"]
# Bead brightness in counts, chosen to sit inside premessa's default gate
# once arcsinh-transformed with a cofactor of 5.
BEAD_LEVELS = {
    "Ce140Di": 220.0,
    "Eu151Di": 180.0,
    "Eu153Di": 260.0,
    "Ho165Di": 150.0,
    "Lu175Di": 200.0,
}
MARKER_CHANNELS = ["Nd142Di", "Sm147Di", "Er168Di"]
DNA_CHANNELS = ["Ir191Di", "Ir193Di"]
# Fluidigm Cell-ID 20-Plex Pd: six palladium channels, three positive each.
BARCODE_CHANNELS = ["Pd102Di", "Pd104Di", "Pd105Di", "Pd106Di", "Pd108Di", "Pd110Di"]


def drift_factor(t, t_max, drift=0.45):
    """Multiplicative sensitivity decay over the acquisition, 1 down to 1-drift."""
    return 1.0 - drift * (t / t_max)


def make_events(n=40_000, bead_fraction=0.06, duration=600_000.0, drift=0.45, seed=0):
    """Interleaved cell and bead events, in acquisition order, with drift applied.

    Parameters
    ----------
    n
        Total number of events.
    bead_fraction
        Fraction of events that are beads.
    duration
        Length of the run in the file's time units.
    drift
        Fractional loss of sensitivity from the start of the run to the end.
    seed
        Seed for the random number generator.

    Returns
    -------
    ndarray
        ``(n, len(CHANNELS))`` float32 matrix, sorted by time.
    """
    rng = np.random.default_rng(seed)
    index = {c: i for i, c in enumerate(CHANNELS)}
    events = np.zeros((n, len(CHANNELS)), dtype=np.float64)

    time = np.sort(rng.uniform(0.0, duration, n))
    events[:, index["Time"]] = time
    events[:, index["Event_length"]] = rng.normal(20.0, 3.0, n).clip(1.0)

    is_bead = rng.random(n) < bead_fraction
    n_bead = int(is_bead.sum())

    # Beads: bright and tight in the bead channels, no DNA, no markers.
    for channel, level in BEAD_LEVELS.items():
        events[is_bead, index[channel]] = level * rng.lognormal(0.0, 0.08, n_bead)
    for channel in DNA_CHANNELS + MARKER_CHANNELS:
        events[is_bead, index[channel]] = rng.gamma(1.0, 0.4, n_bead)

    # Cells: DNA positive, a marker each, only background in the bead channels.
    n_cell = n - n_bead
    for channel in DNA_CHANNELS:
        events[~is_bead, index[channel]] = 400.0 * rng.lognormal(0.0, 0.25, n_cell)
    for k, channel in enumerate(MARKER_CHANNELS):
        positive = rng.random(n_cell) < (0.35, 0.25, 0.2)[k]
        signal = np.where(positive, 120.0 * rng.lognormal(0.0, 0.5, n_cell), 0.0)
        events[~is_bead, index[channel]] = signal + rng.gamma(1.0, 1.5, n_cell)
    for channel in BEAD_CHANNELS:
        events[~is_bead, index[channel]] = rng.gamma(1.0, 1.2, n_cell)

    # Sensitivity decays over the run, beads and cells alike.
    mass = [i for c, i in index.items() if c not in ("Time", "Event_length")]
    events[:, mass] *= drift_factor(time, duration, drift)[:, None]
    return events.astype(np.float32)


def make_barcoded_events(n=60_000, n_barcodes=20, duration=600_000.0, drift=0.45, seed=3):
    """Cells carrying a 3-of-6 palladium barcode, plus beads and drift.

    Parameters
    ----------
    n
        Total number of events.
    n_barcodes
        Barcodes in use, of the 20 the 3-of-6 scheme allows.
    duration
        Length of the run in the file's time units.
    drift
        Fractional loss of sensitivity over the run.
    seed
        Seed for the random number generator.

    Returns
    -------
    tuple
        ``(events, channels, markers, truth)`` -- the matrix, its channel and
        marker names, and the barcode each event really carries (``"0"`` for
        beads and for the deliberate doublets).
    """
    from itertools import combinations

    rng = np.random.default_rng(seed)
    channels = CHANNELS[:-2] + BARCODE_CHANNELS + CHANNELS[-2:]
    markers = MARKERS[:-2] + [""] * len(BARCODE_CHANNELS) + MARKERS[-2:]
    index = {c: i for i, c in enumerate(channels)}
    events = np.zeros((n, len(channels)), dtype=np.float64)

    time = np.sort(rng.uniform(0.0, duration, n))
    events[:, index["Time"]] = time
    events[:, index["Event_length"]] = rng.normal(20.0, 3.0, n).clip(1.0)

    schemes = list(combinations(BARCODE_CHANNELS, 3))[:n_barcodes]
    ids = [f"A{i + 1}" for i in range(len(schemes))]

    kind = rng.choice(["cell", "bead", "doublet"], n, p=[0.88, 0.07, 0.05])
    truth = np.full(n, "0", dtype=object)
    assigned = rng.integers(0, len(schemes), n)

    for channel, level in BEAD_LEVELS.items():
        is_bead = kind == "bead"
        events[is_bead, index[channel]] = level * rng.lognormal(0.0, 0.08, int(is_bead.sum()))
    for channel in DNA_CHANNELS + MARKER_CHANNELS + BARCODE_CHANNELS:
        is_bead = kind == "bead"
        events[is_bead, index[channel]] = rng.gamma(1.0, 0.4, int(is_bead.sum()))

    not_bead = kind != "bead"
    n_rest = int(not_bead.sum())
    for channel in DNA_CHANNELS:
        events[not_bead, index[channel]] = 400.0 * rng.lognormal(0.0, 0.25, n_rest)
    for k, channel in enumerate(MARKER_CHANNELS):
        positive = rng.random(n_rest) < (0.35, 0.25, 0.2)[k]
        events[not_bead, index[channel]] = np.where(
            positive, 120.0 * rng.lognormal(0.0, 0.5, n_rest), 0.0
        ) + rng.gamma(1.0, 1.5, n_rest)
    for channel in BEAD_CHANNELS:
        events[not_bead, index[channel]] = rng.gamma(1.0, 1.2, n_rest)

    # Barcode channels: bright where the scheme says positive, background else.
    for row in np.flatnonzero(not_bead):
        positives = schemes[assigned[row]]
        for channel in BARCODE_CHANNELS:
            if channel in positives:
                events[row, index[channel]] = 300.0 * rng.lognormal(0.0, 0.25)
            else:
                events[row, index[channel]] = rng.gamma(1.0, 1.5)
        truth[row] = ids[assigned[row]]

    # Doublets carry two barcodes at once, so they should end up unassigned.
    for row in np.flatnonzero(kind == "doublet"):
        other = schemes[int(rng.integers(0, len(schemes)))]
        for channel in other:
            events[row, index[channel]] = 300.0 * rng.lognormal(0.0, 0.25)
        events[row, index["Event_length"]] *= 1.8
        truth[row] = "0"

    mass = [i for c, i in index.items() if c not in ("Time", "Event_length")]
    events[:, mass] *= drift_factor(time, duration, drift)[:, None]
    return events.astype(np.float32), channels, markers, truth


def write_barcoded(path="demo_barcoded.fcs", n=60_000):
    """Write the barcoded file and return the true barcode of every event.

    Parameters
    ----------
    path
        FCS file to write.
    n
        Number of events.

    Returns
    -------
    ndarray
        The barcode each event really carries.
    """
    events, channels, markers, truth = make_barcoded_events(n)
    meta = {"$CYT": "cytopy demo CyTOF", "$TIMESTEP": "0.001", "$SRC": Path(path).stem}
    with open(path, "wb") as fh:
        flowio.create_fcs(fh, events.flatten().tolist(), channels, markers, meta)
    print(f"wrote {path}: {events.shape[0]} events x {len(channels)} channels")
    return truth


def main(path="demo_cytof.fcs", n=40_000):
    """Write the synthetic mass cytometry file.

    Parameters
    ----------
    path
        FCS file to write.
    n
        Number of events.
    """
    events = make_events(n)
    meta = {"$CYT": "cytopy demo CyTOF", "$TIMESTEP": "0.001", "$SRC": Path(path).stem}
    with open(path, "wb") as fh:
        flowio.create_fcs(fh, events.flatten().tolist(), CHANNELS, MARKERS, meta)
    print(f"wrote {path}: {events.shape[0]} events x {len(CHANNELS)} channels")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["demo_cytof.fcs"]))
