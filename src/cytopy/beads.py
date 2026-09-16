"""Bead-based normalisation of mass cytometry data (Finck et al., 2013).

Signal drift over an acquisition is corrected by spiking in metal-containing
polystyrene beads: because the beads themselves do not change, the change in
*their* intensity over time is the instrument's drift, and every channel can be
scaled back onto a fixed baseline.

The arithmetic here follows the two reference implementations, which agree with
each other on all of it:

* CATALYST's ``normCytof`` (Crowell et al.), and
* premessa's ``normalize_folder`` (Parker Institute).

For each bead event the bead-channel intensities are smoothed over time with a
running median, and the correction factor is the slope of the no-intercept
least-squares fit of the baseline against those smoothed intensities,
``sum(b * B) / sum(b * b)``. Slopes are interpolated linearly across the
non-bead events and every mass channel is multiplied by them.

Where the two differ, the defaults here follow premessa, because the bead gate
is premessa's: a rectangle per bead channel, drawn on arcsinh-transformed data,
that you are meant to look at and adjust. See :func:`normalise_beads` for the
list of differences and how to switch.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from itertools import pairwise

import anndata as ad
import numpy as np
import pandas as pd

from .transforms import channel_index

__all__ = [
    "BEAD_PANELS",
    "bead_baseline",
    "bead_channels",
    "bead_distance",
    "bead_gates",
    "dna_channel",
    "gate_beads",
    "mass_channels",
    "normalise_beads",
    "plot_beads_over_time",
]

#: Bead sets, by the isotopes each one is identified on. ``"dvs"`` is
#: CATALYST's name for the Fluidigm/Standard BioTools EQ four-element beads.
BEAD_PANELS: dict[str, tuple[str, ...]] = {
    "fluidigm": ("Ce140", "Eu151", "Eu153", "Ho165", "Lu175"),
    "beta": ("La139", "Pr141", "Tb159", "Tm169", "Lu175"),
    "xt": ("Y89", "In115", "Ce140", "Tb159", "Lu175", "Bi209"),
}
BEAD_PANELS["dvs"] = BEAD_PANELS["fluidigm"]

#: DNA intercalator channels, most preferred first.
DNA_ISOTOPES = ("Ir193", "Ir191")

#: premessa's starting gate for every bead channel, on ``asinh(x / 5)``:
#: bead channel between 2 and 5, DNA between -1 and 2.
DEFAULT_GATE = {"x": (2.0, 5.0), "y": (-1.0, 2.0)}

#: Cofactor of the arcsinh the bead gate is drawn on. 5 is the mass cytometry
#: convention and what premessa hard-codes.
BEAD_COFACTOR = 5.0

#: Rows per block when a pass has to touch every event. Big enough that the
#: per-block overhead is invisible, small enough that the temporaries stay in
#: cache-friendly megabytes on a run of millions of events.
_CHUNK = 1 << 20

# Fluidigm writes mass channels as "Ce140Di"; the housekeeping columns that
# describe the event rather than measure it are never scaled.
_MASS_RE = re.compile(r"(Di|Dd)$", re.IGNORECASE)
_HOUSEKEEPING_RE = re.compile(
    r"^(time|event_?length|center|offset|width|residual|beaddist)$", re.IGNORECASE
)


def _isotope_pattern(token: str) -> re.Pattern[str]:
    """Match an isotope by element+mass (``Ce140``), either order, or mass alone.

    Instruments and conversion tools disagree about the spelling: ``Ce140Di``,
    ``140Ce``, ``Ce140_EQBeads`` and a bare ``140`` all name the same channel.
    """
    element = re.match(r"([A-Za-z]*)(\d+)", token.strip())
    if element is None:
        return re.compile(rf"(?<!\d){re.escape(token)}(?!\d)", re.IGNORECASE)
    name, mass = element.groups()
    if not name:
        return re.compile(rf"(?<!\d){mass}(?!\d)")
    return re.compile(rf"(?<!\d)({name}{mass}|{mass}{name})(?!\d)", re.IGNORECASE)


def _channel_names(adata: ad.AnnData) -> list[str]:
    """The ``$PnN`` detector names, falling back to the var index."""
    if "channel" in adata.var:
        return [str(v) for v in adata.var["channel"]]
    return [str(v) for v in adata.var_names]


def _resolve_isotope(adata: ad.AnnData, token: str) -> str | None:
    """The var_name of the channel measuring ``token``, or None."""
    pattern = _isotope_pattern(token)
    for name, channel in zip(adata.var_names, _channel_names(adata)):
        if pattern.search(channel) or pattern.search(str(name)):
            return str(name)
    return None


def bead_channels(adata: ad.AnnData, beads: str | Sequence[str | int] = "fluidigm") -> list[str]:
    """Resolve a bead set to the channels that measure it.

    Parameters
    ----------
    adata
        Mass cytometry AnnData, e.g. from :func:`~cytopy.read_fcs`.
    beads
        A key of :data:`BEAD_PANELS` (``"fluidigm"`` / ``"dvs"``, ``"beta"``,
        ``"xt"``), or your own sequence of isotopes (``["Ce140", "Eu151"]``),
        masses (``[140, 151]``) or channel names.

    Returns
    -------
    list of str
        var_names, in the order the isotopes were given.

    Raises
    ------
    KeyError
        If any isotope has no channel, naming the ones that are missing. This
        is fatal rather than a warning: a bead set with a channel silently
        dropped still normalises, just wrongly.
    """
    if isinstance(beads, str):
        try:
            tokens: Sequence[str | int] = BEAD_PANELS[beads.lower()]
        except KeyError:
            raise KeyError(
                f"unknown bead set {beads!r}; use one of {sorted(BEAD_PANELS)} "
                "or pass the isotopes yourself"
            ) from None
    else:
        tokens = list(beads)

    found, missing = [], []
    for token in tokens:
        name = _resolve_isotope(adata, str(token))
        if name is None:
            missing.append(str(token))
        elif name not in found:
            found.append(name)
    if missing:
        raise KeyError(
            f"no channel for bead isotope(s) {missing}; available: {_channel_names(adata)}"
        )
    return found


def dna_channel(adata: ad.AnnData, dna: str | int | Sequence[str | int] | None = None) -> str:
    """Resolve the DNA intercalator channel used as the gate's second axis.

    Parameters
    ----------
    adata
        Mass cytometry AnnData.
    dna
        Isotope, mass, or channel name. ``None`` tries Ir193 then Ir191, which
        is what premessa and CATALYST look for.

    Returns
    -------
    str
        var_name of the DNA channel.

    Raises
    ------
    KeyError
        If no DNA channel is found.
    """
    tokens = DNA_ISOTOPES if dna is None else ([dna] if isinstance(dna, str | int) else list(dna))
    for token in tokens:
        name = _resolve_isotope(adata, str(token))
        if name is not None:
            return name
    raise KeyError(f"no DNA channel matching {list(tokens)}; available: {_channel_names(adata)}")


def mass_channels(adata: ad.AnnData, channels: Sequence[str] | None = None) -> list[str]:
    """The channels normalisation should scale: the mass channels.

    Time, event length, and the pushes that describe an event rather than
    measure it are left alone, matching premessa's ``Di$|Dd$`` rule.

    Parameters
    ----------
    adata
        Mass cytometry AnnData.
    channels
        Explicit list, resolved and returned as-is. ``None`` detects them.

    Returns
    -------
    list of str
        var_names to scale, in file order.
    """
    if channels is not None:
        return [str(adata.var_names[channel_index(adata, c)]) for c in channels]
    names = list(adata.var_names)
    detectors = _channel_names(adata)
    tagged = [n for n, d in zip(names, detectors) if _MASS_RE.search(d)]
    if tagged:
        return tagged
    # Not Fluidigm-style names: keep everything that is not housekeeping.
    kinds = adata.var.get("kind", None)
    out = []
    for j, (name, detector) in enumerate(zip(names, detectors)):
        if _HOUSEKEEPING_RE.match(detector.strip()):
            continue
        if kinds is not None and str(kinds.iloc[j]) == "time":
            continue
        out.append(name)
    return out


def _time_column(adata: ad.AnnData) -> np.ndarray:
    """Acquisition time, in the file's own units."""
    if "kind" in adata.var:
        hits = np.flatnonzero(adata.var["kind"].to_numpy() == "time")
        if hits.size:
            return np.asarray(adata.X[:, hits[0]], dtype=np.float64).ravel()
    try:
        j = channel_index(adata, "Time")
    except KeyError:
        raise KeyError(
            "no Time channel; bead normalisation needs one to order and "
            "interpolate the correction factors"
        ) from None
    return np.asarray(adata.X[:, j], dtype=np.float64).ravel()


def _sample_groups(adata: ad.AnnData, sample_key: str | None) -> list[tuple[str, np.ndarray]]:
    """``(label, row indices)`` per acquisition, in file order.

    Grouping goes through the categorical's integer codes. Reaching for
    ``obs[key].astype(str)`` instead would build one Python string per event,
    which on a run of millions costs more than the normalisation itself.
    """
    if sample_key is None or sample_key not in adata.obs:
        return [("", np.arange(adata.n_obs))]
    column = adata.obs[sample_key]
    if isinstance(column.dtype, pd.CategoricalDtype):
        codes = np.asarray(column.cat.codes)
        categories = [str(c) for c in column.cat.categories]
    else:
        codes, values = pd.factorize(column.to_numpy())
        categories = [str(c) for c in values]

    order = np.argsort(codes, kind="stable")  # stable: rows stay in file order
    ordered = codes[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    bounds = np.r_[starts, codes.size]
    groups = []
    for start, stop in pairwise(bounds):
        code = int(ordered[start])
        label = categories[code] if code >= 0 else "<missing>"
        groups.append((label, order[start:stop]))
    return groups


def bead_gates(
    adata: ad.AnnData,
    beads: str | Sequence[str | int] = "fluidigm",
    gates: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> dict[str, dict[str, tuple[float, float]]]:
    """The bead gate as a rectangle per bead channel, filling in the defaults.

    Each entry is ``{"x": (lo, hi), "y": (lo, hi)}`` where *x* is the bead
    channel and *y* the DNA channel, both on ``asinh(value / 5)``. The default
    is premessa's starting gate, :data:`DEFAULT_GATE` — beads are bright in the
    bead channel and dim in DNA. It is a starting point, not an answer: look at
    it against your own data and move it.

    Parameters
    ----------
    adata
        Mass cytometry AnnData.
    beads
        Bead set, as for :func:`bead_channels`.
    gates
        Overrides, keyed by bead channel (any alias :func:`~cytopy.channel_index`
        accepts). A partial entry keeps the default for the other axis, so
        ``{"Ce140Di": {"x": (2.5, 5)}}`` moves one edge and leaves the rest.

    Returns
    -------
    dict
        One entry per bead channel, keyed by var_name.

    Raises
    ------
    KeyError
        If an override names a channel that is not in the bead set.
    """
    names = bead_channels(adata, beads)
    out = {name: {"x": tuple(DEFAULT_GATE["x"]), "y": tuple(DEFAULT_GATE["y"])} for name in names}
    for key, gate in (gates or {}).items():
        resolved = str(adata.var_names[channel_index(adata, key)])
        if resolved not in out:
            raise KeyError(f"gate for {key!r} is not a bead channel; bead channels are {names}")
        for axis in ("x", "y"):
            if axis in gate:
                lo, hi = (float(v) for v in gate[axis])
                out[resolved][axis] = (lo, hi)
    return out


def gate_beads(
    adata: ad.AnnData,
    *,
    beads: str | Sequence[str | int] = "fluidigm",
    dna: str | int | Sequence[str | int] | None = None,
    gates: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    cofactor: float = BEAD_COFACTOR,
    trim: float | None = None,
    key_added: str | None = "bead",
) -> np.ndarray:
    """Identify bead events with premessa's rectangular gates.

    An event is a bead when it falls inside the rectangle for *every* bead
    channel — bright in the bead channel, dim in DNA — with the comparison made
    on ``asinh(value / cofactor)`` and the edges exclusive, as in premessa's
    ``identify_beads``.

    Parameters
    ----------
    adata
        Mass cytometry AnnData holding raw counts in ``.X``.
    beads
        Bead set, as for :func:`bead_channels`.
    dna
        DNA channel, as for :func:`dna_channel`.
    gates
        Gate overrides, as for :func:`bead_gates`. ``None`` uses the defaults,
        which you should look at before trusting.
    cofactor
        Arcsinh cofactor the gate coordinates are in. Leave at 5 unless you
        know your gates were drawn on something else.
    trim
        Optional extra filter, off by default: drop bead events lying more than
        ``trim`` MADs from the bead median in any bead channel, which is how
        CATALYST removes bead-bead and bead-cell doublets (it uses ``5``).
    key_added
        ``obs`` column to write the mask to, and the key under which the gates
        and counts are recorded in ``adata.uns['cytopy']['beads']``. ``None``
        writes nothing and only returns the mask.

    Returns
    -------
    ndarray
        Boolean mask over events, ``True`` for beads.

    Raises
    ------
    ValueError
        If the gate selects no events at all.
    """
    resolved = bead_gates(adata, beads, gates)
    dna_name = dna_channel(adata, dna)
    dna_col = channel_index(adata, dna_name)
    bead_cols = [channel_index(adata, n) for n in resolved]

    # One pass in row blocks: an event is a bead only if it is inside every
    # channel's rectangle, so the columns are combined block by block rather
    # than each one being expanded over the whole run.
    mask = np.empty(adata.n_obs, dtype=bool)
    for start in range(0, adata.n_obs, _CHUNK):
        stop = min(start + _CHUNK, adata.n_obs)
        dna_values = np.arcsinh(
            np.asarray(adata.X[start:stop, dna_col], dtype=np.float64).ravel() / cofactor
        )
        block = np.ones(stop - start, dtype=bool)
        for col, gate in zip(bead_cols, resolved.values()):
            values = np.arcsinh(
                np.asarray(adata.X[start:stop, col], dtype=np.float64).ravel() / cofactor
            )
            (x_lo, x_hi), (y_lo, y_hi) = gate["x"], gate["y"]
            block &= (values > x_lo) & (values < x_hi) & (dna_values > y_lo) & (dna_values < y_hi)
        mask[start:stop] = block

    if not mask.any():
        raise ValueError(
            "the bead gate selects no events; the default gate assumes "
            f"asinh(x / {cofactor:g}) — plot the bead channels against {dna_name} and move it"
        )

    if trim is not None:
        inside = np.arcsinh(
            np.asarray(adata.X[np.ix_(np.flatnonzero(mask), bead_cols)], dtype=np.float64)
            / cofactor
        )
        median = np.median(inside, axis=0)
        mad = np.median(np.abs(inside - median), axis=0) * 1.4826 * float(trim)
        keep = ~np.any(np.abs(inside - median) > mad, axis=1)
        mask[np.flatnonzero(mask)] = keep

    if key_added is not None:
        adata.obs[key_added] = mask
        info = adata.uns.setdefault("cytopy", {}).setdefault("beads", {})
        info["gates"] = {k: {a: list(v) for a, v in g.items()} for k, g in resolved.items()}
        info["channels"] = list(resolved)
        info["dna"] = dna_name
        info["cofactor"] = float(cofactor)
        info["key"] = key_added
        info["n_beads"] = int(mask.sum())
    return mask


# --------------------------------------------------------------------------
# R-compatible running median
# --------------------------------------------------------------------------
def _med3(a: float, b: float, c: float) -> float:
    return max(min(a, b), min(max(a, b), c))


def _smooth_ends(y: np.ndarray, k: int) -> np.ndarray:
    """R's ``stats::smoothEnds``: Tukey end rule for a running median.

    Positions the median window cannot cover are filled with medians of
    progressively shorter odd windows, and the two extreme points with a
    median-of-three against a linear extrapolation.
    """
    n = y.size
    if k < 3 or n < 3:
        return y.copy()
    half = k // 2
    out = y.copy()
    for i in range(1, half):
        j = 2 * i + 1
        if j > n:
            break
        out[i] = np.median(y[:j])
        out[n - i - 1] = np.median(y[n - j :])
    out[0] = _med3(y[0], out[1], 3.0 * out[1] - 2.0 * out[2])
    out[n - 1] = _med3(y[n - 1], out[n - 2], 3.0 * out[n - 2] - 2.0 * out[n - 3])
    return out


def _runmed(x: np.ndarray, k: int, endrule: str = "median") -> np.ndarray:
    """Running median of width ``k``, matching R's ``stats::runmed``.

    ``endrule`` is what happens to the ``k // 2`` points at each end that no
    full window covers: ``"median"`` (R's default, premessa) applies the Tukey
    end rule, ``"constant"`` (CATALYST) repeats the first and last full-window
    value, ``"keep"`` leaves the input values there.
    """
    from scipy.ndimage import median_filter

    x = np.asarray(x, dtype=np.float64)
    n = x.size
    k = int(k)
    if k % 2 == 0:
        k += 1
    if k > n:  # R shrinks the window to the largest odd width that fits
        k = n if n % 2 else max(n - 1, 1)
    if k <= 1 or n == 0:
        return x.copy()

    half = k // 2
    out = median_filter(x, size=k, mode="nearest")
    if endrule == "constant":
        out[:half] = out[half]
        out[n - half :] = out[n - half - 1]
        return out
    out[:half] = x[:half]
    out[n - half :] = x[n - half :]
    if endrule == "median":
        return _smooth_ends(out, k)
    if endrule == "keep":
        return out
    raise ValueError(f"endrule must be 'median', 'constant' or 'keep', not {endrule!r}")


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------
def bead_baseline(
    adata: ad.AnnData,
    *,
    beads: str | Sequence[str | int] = "fluidigm",
    bead_key: str = "bead",
    statistic: str = "median",
    layer: str | None = None,
) -> pd.Series:
    """The bead intensities to normalise onto.

    Parameters
    ----------
    adata
        Mass cytometry AnnData with a bead mask already in ``obs[bead_key]``.
    beads
        Bead set, as for :func:`bead_channels`.
    bead_key
        ``obs`` column marking bead events.
    statistic
        ``"median"`` (premessa) or ``"mean"`` (CATALYST) over the bead events.
    layer
        Layer to measure. ``None`` uses ``.X``; it must hold raw counts, since
        the fit is a linear one.

    Returns
    -------
    Series
        Baseline intensity per bead channel, indexed by var_name. Pass it to
        :func:`normalise_beads` to put several acquisitions on one scale.

    Raises
    ------
    KeyError
        If ``bead_key`` is not an ``obs`` column.
    ValueError
        If no events are marked as beads, or ``statistic`` is unknown.
    """
    if bead_key not in adata.obs:
        raise KeyError(f"no obs column {bead_key!r}; run cytopy.gate_beads first")
    names = bead_channels(adata, beads)
    mask = np.asarray(adata.obs[bead_key], dtype=bool)
    if not mask.any():
        raise ValueError(f"obs[{bead_key!r}] marks no events as beads")
    matrix = adata.X if layer is None else adata.layers[layer]
    # Index rows and columns together: pulling whole columns first would
    # materialise one float64 array per channel over every event in the file.
    counts = np.asarray(
        matrix[np.ix_(np.flatnonzero(mask), [channel_index(adata, n) for n in names])],
        dtype=np.float64,
    )
    if statistic == "median":
        values = np.median(counts, axis=0)
    elif statistic == "mean":
        values = np.mean(counts, axis=0)
    else:
        raise ValueError(f"statistic must be 'median' or 'mean', not {statistic!r}")
    return pd.Series(values, index=pd.Index(names, name="channel"), name="baseline")


def _slopes_for_sample(
    bead_counts: np.ndarray,
    bead_times: np.ndarray,
    times: np.ndarray,
    baseline: np.ndarray,
    k: int,
    endrule: str,
) -> np.ndarray:
    """Per-event correction factors for one acquisition, over all its events.

    ``bead_counts`` is the bead events' bead channels only, in time order --
    a few hundred thousand rows at most, however many events the file has.
    """
    smoothed = np.empty_like(bead_counts)
    for j in range(bead_counts.shape[1]):
        smoothed[:, j] = _runmed(bead_counts[:, j], k, endrule)

    # Slope of the no-intercept least-squares fit of baseline on smoothed
    # intensities: sum(b * B) / sum(b * b), one per bead event.
    denominator = np.sum(smoothed**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.where(denominator > 0, (smoothed @ baseline) / denominator, 1.0)

    # Linear interpolation, held constant outside the bead time range
    # (R's approx(..., rule = 2)).
    return np.interp(times, bead_times, slopes)


def normalise_beads(
    adata: ad.AnnData,
    *,
    beads: str | Sequence[str | int] = "fluidigm",
    dna: str | int | Sequence[str | int] | None = None,
    gates: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    bead_key: str = "bead",
    baseline: pd.Series | Mapping[str, float] | Sequence[float] | None = None,
    statistic: str = "median",
    k: int = 201,
    endrule: str = "median",
    channels: Sequence[str] | None = None,
    sample_key: str | None = "sample",
    trim: float | None = None,
    layer: str | None = None,
    key_added: str = "normalised",
    stash_curves: int | None = 2000,
    copy: bool = False,
) -> ad.AnnData:
    """Bead-normalise mass cytometry data (Finck et al., 2013).

    Bead events are smoothed over time with a running median; the correction
    factor at each of them is the slope of the no-intercept least-squares fit
    of the baseline against the smoothed intensities, ``sum(b * B) / sum(b * b)``;
    those slopes are interpolated linearly over the whole acquisition and every
    mass channel is multiplied by them.

    Defaults follow premessa. CATALYST's ``normCytof`` differs in three places,
    all reachable from here: it averages the baseline (``statistic="mean"``),
    smooths over a window of 500 (``k=500``), and repeats the end values rather
    than applying the Tukey end rule (``endrule="constant"``). Its automatic
    bead gate has no equivalent — the gate here is premessa's, meant to be
    looked at and adjusted; ``trim`` is CATALYST's doublet filter.

    Parameters
    ----------
    adata
        Mass cytometry AnnData with raw counts, a Time channel, and one row per
        event.
    beads
        Bead set, as for :func:`bead_channels`.
    dna
        DNA channel for the gate, as for :func:`dna_channel`.
    gates
        Gate overrides, as for :func:`bead_gates`. Ignored when
        ``obs[bead_key]`` already exists.
    bead_key
        ``obs`` column holding the bead mask. Used when present — so a gate
        drawn in the viewer, or one from a previous :func:`gate_beads` call,
        wins over ``gates`` — and written by gating here when not.
    baseline
        What to normalise onto: a Series or mapping of bead channel to
        intensity (from :func:`bead_baseline` on a reference acquisition), or
        ``None`` to use this data's own bead events, which is what both
        references do by default.
    statistic
        ``"median"`` (premessa) or ``"mean"`` (CATALYST) for the baseline.
        Ignored when ``baseline`` is given.
    k
        Width of the running median window over bead events, in events. Forced
        odd, and shrunk if there are fewer bead events than that. premessa uses
        201, CATALYST 500.
    endrule
        How the ``k // 2`` events at each end are smoothed: ``"median"``
        (premessa, R's default), ``"constant"`` (CATALYST) or ``"keep"``.
    channels
        Channels to scale. Defaults to :func:`mass_channels`, which leaves
        Time, event length and the other housekeeping columns alone.
    sample_key
        ``obs`` column separating acquisitions. Each is smoothed and
        interpolated on its own clock, since Time restarts with every file,
        while the baseline is shared across all of them. ``None`` treats the
        object as one acquisition.
    trim
        CATALYST-style doublet filter, passed to :func:`gate_beads`. Ignored
        when ``obs[bead_key]`` already exists.
    layer
        Input layer of raw counts. ``None`` uses ``adata.X``.
    key_added
        Layer to write the normalised counts to. ``None`` overwrites
        ``adata.X`` instead, which halves the peak memory on a large run.
    stash_curves
        Points per curve to keep of the smoothed bead intensities, before and
        after, in ``uns``. They are what lets :func:`plot_beads_over_time` and
        :func:`~cytopy.report` still draw the figure once the bead events
        themselves have been removed -- which is the normal order of a
        pipeline. ``None`` skips it and saves a second pass of smoothing.
    copy
        Return a modified copy instead of writing to ``adata`` in place.

    Returns
    -------
    AnnData
        The annotated object, with normalised counts in
        ``adata.layers[key_added]`` (or ``adata.X``), the per-event correction factor in
        ``adata.obs['bead_slope']``, the bead mask in ``adata.obs[bead_key]``,
        and the baseline, gates and per-sample bead counts in
        ``adata.uns['cytopy']['beads']``.

    Raises
    ------
    ValueError
        If a sample has no bead events, or the baseline is not positive.
    KeyError
        If the bead, DNA or Time channels cannot be found.
    """
    adata = adata.copy() if copy else adata
    names = bead_channels(adata, beads)

    if bead_key in adata.obs:
        mask = np.asarray(adata.obs[bead_key], dtype=bool)
    else:
        mask = gate_beads(adata, beads=beads, dna=dna, gates=gates, trim=trim, key_added=bead_key)

    if baseline is None:
        base = bead_baseline(
            adata, beads=beads, bead_key=bead_key, statistic=statistic, layer=layer
        )
    elif isinstance(baseline, pd.Series):
        base = baseline
    elif isinstance(baseline, Mapping):
        base = pd.Series({k_: float(v) for k_, v in baseline.items()})
    else:
        base = pd.Series(np.asarray(baseline, dtype=float), index=pd.Index(names))
    # Align the baseline to this object's channels, by whatever alias it uses.
    lookup = {str(adata.var_names[channel_index(adata, k_)]): float(v) for k_, v in base.items()}
    missing = [n for n in names if n not in lookup]
    if missing:
        raise KeyError(f"baseline has no value for bead channel(s) {missing}")
    base_values = np.array([lookup[n] for n in names])
    if not np.all(base_values > 0):
        raise ValueError(f"baseline must be positive, got {dict(zip(names, base_values))}")

    matrix = adata.X if layer is None else adata.layers[layer]
    times = _time_column(adata)
    bead_cols = [channel_index(adata, n) for n in names]
    scale_cols = [channel_index(adata, n) for n in mass_channels(adata, channels)]

    slopes = np.ones(adata.n_obs)
    per_sample: dict[str, dict[str, float]] = {}
    for label, rows in _sample_groups(adata, sample_key):
        if not mask[rows].any():
            raise ValueError(
                f"sample {label!r} has no bead events; it cannot be normalised. "
                "Adjust the gate, or drop the sample."
            )
        # Only the beads' own channels are needed to fit the slopes, so this
        # stays small no matter how many events or channels the file has.
        beads_here = rows[mask[rows]]
        beads_here = beads_here[np.argsort(times[beads_here], kind="stable")]
        bead_counts = np.asarray(matrix[np.ix_(beads_here, bead_cols)], dtype=np.float64)
        sample_slopes = _slopes_for_sample(
            bead_counts, times[beads_here], times[rows], base_values, k, endrule
        )
        slopes[rows] = sample_slopes
        per_sample[str(label)] = {
            "n_beads": int(mask[rows].sum()),
            "median_slope": float(np.median(sample_slopes)),
            "min_slope": float(sample_slopes.min()),
            "max_slope": float(sample_slopes.max()),
        }

    # Scale a column at a time, into an array already at the storage dtype: a
    # float64 copy of the whole matrix costs eight bytes per value per copy,
    # which is gigabytes on a run of a few million events.
    dtype = adata.X.dtype
    out = np.array(matrix, dtype=dtype, copy=True)
    for j in scale_cols:
        out[:, j] = (np.asarray(matrix[:, j], dtype=np.float64) * slopes).astype(dtype)
    if key_added is None:
        adata.X = out
    else:
        adata.layers[key_added] = out
    adata.obs["bead_slope"] = slopes

    if stash_curves:
        curves = _bead_curves(
            adata,
            matrix,
            out,
            mask,
            times,
            bead_cols,
            names,
            k,
            endrule,
            sample_key,
            int(stash_curves),
        )
    info = adata.uns.setdefault("cytopy", {}).setdefault("beads", {})
    if stash_curves:
        info["curves"] = curves
    info.update(
        {
            "normalised_layer": key_added,
            "channels": names,
            "baseline": {n: float(v) for n, v in zip(names, base_values)},
            "statistic": statistic if baseline is None else "given",
            "k": int(k),
            "endrule": endrule,
            "key": bead_key,
            "samples": per_sample,
            "scaled_channels": [str(adata.var_names[j]) for j in scale_cols],
        }
    )
    return adata


def bead_distance(
    adata: ad.AnnData,
    *,
    beads: str | Sequence[str | int] = "fluidigm",
    bead_key: str = "bead",
    cofactor: float = BEAD_COFACTOR,
    layer: str | None = None,
    key_added: str | None = "bead_distance",
) -> np.ndarray:
    """Distance of every event from the centre of the bead population.

    premessa's ``beadDist``: the square root of the Mahalanobis distance in the
    space of the arcsinh-transformed bead channels, with the centre and
    covariance taken from the gated beads. Events close to the beads are beads
    or bead-cell doublets; thresholding on it is how premessa removes them
    after normalisation, which catches doublets the rectangular gate lets
    through.

    Parameters
    ----------
    adata
        Mass cytometry AnnData with a bead mask in ``obs[bead_key]``.
    beads
        Bead set, as for :func:`bead_channels`.
    bead_key
        ``obs`` column marking bead events.
    cofactor
        Arcsinh cofactor the distance is measured on.
    layer
        Layer to measure. ``None`` uses ``.X``. Pass the normalised layer to
        match premessa, which computes this after normalising.
    key_added
        ``obs`` column to write to, or ``None`` to only return the values.

    Returns
    -------
    ndarray
        Distance per event. Remove beads with e.g. ``adata[dist > 3]``.

    Raises
    ------
    KeyError
        If ``bead_key`` is not an ``obs`` column.
    ValueError
        If the bead population is too small to estimate a covariance from.
    """
    from scipy.linalg import pinvh

    if bead_key not in adata.obs:
        raise KeyError(f"no obs column {bead_key!r}; run cytopy.gate_beads first")
    names = bead_channels(adata, beads)
    matrix = adata.X if layer is None else adata.layers[layer]
    cols = [channel_index(adata, n) for n in names]
    mask = np.asarray(adata.obs[bead_key], dtype=bool)
    if int(mask.sum()) <= len(names):
        raise ValueError(
            f"only {int(mask.sum())} bead events for {len(names)} channels; "
            "not enough to estimate a covariance"
        )
    population = np.arcsinh(
        np.asarray(matrix[np.ix_(np.flatnonzero(mask), cols)], dtype=np.float64) / cofactor
    )
    centre = population.mean(axis=0)
    # pinvh, not inv: a bead population with a degenerate channel is singular
    # but still has a well-defined distance in the directions that vary.
    precision = pinvh(np.cov(population, rowvar=False))
    del population

    # In row blocks, so peak memory follows the panel width and not the number
    # of events.
    distance = np.empty(adata.n_obs, dtype=np.float64)
    for start in range(0, adata.n_obs, _CHUNK):
        stop = min(start + _CHUNK, adata.n_obs)
        block = np.arcsinh(np.asarray(matrix[start:stop, cols], dtype=np.float64) / cofactor)
        block -= centre
        distance[start:stop] = np.sqrt(
            np.maximum(np.einsum("ij,jk,ik->i", block, precision, block), 0.0)
        )
    if key_added is not None:
        adata.obs[key_added] = distance
    return distance


def plot_beads_over_time(
    adata: ad.AnnData,
    *,
    beads: str | Sequence[str | int] = "fluidigm",
    bead_key: str = "bead",
    layer: str | None = None,
    normalised_layer: str | None = None,
    k: int = 201,
    endrule: str = "median",
    sample_key: str | None = "sample",
    max_points: int = 10_000,
    ax=None,
):
    """Smoothed bead intensities against time, before and after normalisation.

    The diagnostic both references print: flat, overlapping lines mean the
    drift has been taken out; lines that still slope mean the gate is picking
    up something other than beads.

    Parameters
    ----------
    adata
        Mass cytometry AnnData with a bead mask in ``obs[bead_key]``.
    beads
        Bead set, as for :func:`bead_channels`.
    bead_key
        ``obs`` column marking bead events.
    layer
        Layer of raw counts. ``None`` uses ``.X``.
    normalised_layer
        Layer of normalised counts for the lower panel. Defaults to whatever
        :func:`normalise_beads` recorded; ``None`` with nothing recorded draws
        the before panel only.
    k, endrule
        Smoothing, as for :func:`normalise_beads`. Match what you normalised
        with so the two panels are comparable.
    sample_key
        ``obs`` column separating acquisitions; each is smoothed on its own
        clock and drawn end to end.
    max_points
        Most points to draw per line, per sample. The smoothing still runs over
        every bead event -- only the drawing is thinned, by taking every *n*th
        point of the finished curve, which is what keeps this quick on a run of
        millions of events. ``0`` draws all of them.
    ax
        Existing axes to draw on. A one- or two-panel figure is created when
        omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, ready to save or show.

    Raises
    ------
    KeyError
        If ``bead_key`` is not an ``obs`` column.
    """
    import matplotlib.pyplot as plt

    recorded = adata.uns.get("cytopy", {}).get("beads", {})
    live = bead_key in adata.obs and bool(np.any(np.asarray(adata.obs[bead_key], dtype=bool)))
    if not live:
        # The beads have been removed; fall back to the curves normalisation
        # stashed while they were still here.
        if recorded.get("curves"):
            return _plot_stashed_curves(recorded["curves"], max_points, ax)
        raise KeyError(
            f"no bead events in obs[{bead_key!r}] and no stashed curves; run "
            "cytopy.gate_beads, or normalise with stash_curves before removing them"
        )
    names = bead_channels(adata, beads)
    if normalised_layer is None:
        normalised_layer = recorded.get("normalised_layer")
    if normalised_layer is not None and normalised_layer not in adata.layers:
        normalised_layer = None

    panels = [("before", layer)] + ([("after", normalised_layer)] if normalised_layer else [])
    if ax is None:
        fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3 * len(panels)), sharex=True)
        axes = np.atleast_1d(axes)
    else:
        fig, axes = ax.figure, np.atleast_1d(ax)[: len(panels)]

    mask = np.asarray(adata.obs[bead_key], dtype=bool)
    times = _time_column(adata)
    cols = [channel_index(adata, n) for n in names]
    groups = [(label, rows[mask[rows]]) for label, rows in _sample_groups(adata, sample_key)]
    groups = [(label, rows) for label, rows in groups if rows.size]

    colours = plt.get_cmap("tab10")
    for axis, (title, key) in zip(axes, panels):
        matrix = adata.X if key is None else adata.layers[key]
        for first, (_, rows) in enumerate(groups):
            rows = rows[np.argsort(times[rows], kind="stable")]
            block = np.asarray(matrix[np.ix_(rows, cols)], dtype=np.float64)
            step = 1
            if max_points and rows.size > max_points:
                step = int(np.ceil(rows.size / max_points))
            smoothed = np.column_stack(
                [_runmed(block[:, j], k, endrule) for j in range(len(names))]
            )
            for j, name in enumerate(names):
                axis.plot(
                    times[rows][::step],
                    smoothed[::step, j],
                    color=colours(j % 10),
                    lw=1.2,
                    label=name if first == 0 else None,
                )
            # The dashed line is where this channel sits on average in this
            # panel: after normalising they should land on the baseline and
            # stay there, which is the whole point of the figure.
            for j in range(len(names)):
                axis.axhline(
                    float(np.mean(smoothed[:, j])),
                    color=colours(j % 10),
                    lw=0.7,
                    ls="--",
                    alpha=0.8,
                )
        axis.set_title(title, loc="left", fontsize=10)
        axis.set_ylabel("smoothed intensity")
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("time")
    axes[0].legend(fontsize=8, ncol=len(names), frameon=False)
    fig.tight_layout()
    return fig


def remove_beads(
    adata: ad.AnnData,
    *,
    beads: str | Sequence[str | int] = "fluidigm",
    bead_key: str = "bead",
    distance_cutoff: float | None = None,
    layer: str | None = None,
    cofactor: float = BEAD_COFACTOR,
    copy: bool = True,
) -> ad.AnnData:
    """Drop the bead events, and optionally the doublets around them.

    The gate identifies beads; this is the step that acts on it. What went is
    recorded in the filter log, so a report can account for every event
    without the intermediates being kept around.

    Parameters
    ----------
    adata
        Mass cytometry AnnData with a bead mask in ``obs[bead_key]``.
    beads
        Bead set, as for :func:`bead_channels`.
    bead_key
        ``obs`` column marking bead events.
    distance_cutoff
        Also drop events within this Mahalanobis distance of the bead centroid
        -- premessa's ``beadDist`` rule, which catches the bead-cell doublets
        a rectangular gate lets through. ``None`` drops only the gated beads.
        Plot :func:`bead_distance` before choosing one.
    layer
        Layer to measure the distance on. ``None`` uses ``.X``; pass the
        normalised layer to match premessa, which does this after normalising.
    cofactor
        Arcsinh cofactor for the distance.
    copy
        Return a new object rather than a view.

    Returns
    -------
    AnnData
        The events that are neither beads nor, if asked, bead-adjacent.

    Raises
    ------
    KeyError
        If ``bead_key`` is not an ``obs`` column.
    """
    from .filters import filter_events

    if bead_key not in adata.obs:
        raise KeyError(f"no obs column {bead_key!r}; run cytopy.gate_beads first")
    mask = np.asarray(adata.obs[bead_key], dtype=bool)
    keep = ~mask
    reason = f"inside the bead gate ({int(mask.sum()):,} events)"
    params: dict[str, object] = {"bead_key": bead_key}

    if distance_cutoff is not None:
        distance = bead_distance(
            adata, beads=beads, bead_key=bead_key, cofactor=cofactor, layer=layer
        )
        near = distance <= float(distance_cutoff)
        # Report what the distance rule adds, not how many events it covers:
        # most of those are inside the gate already.
        extra = int((near & ~mask).sum())
        keep &= ~near
        reason += f"; {extra:,} more within {distance_cutoff:g} of the bead centroid"
        params["distance_cutoff"] = float(distance_cutoff)

    return filter_events(adata, keep, step="remove beads", reason=reason, params=params, copy=copy)


def plot_bead_gates(
    adata: ad.AnnData,
    *,
    beads: str | Sequence[str | int] = "fluidigm",
    dna: str | int | Sequence[str | int] | None = None,
    bead_key: str = "bead",
    gates: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    cofactor: float = BEAD_COFACTOR,
    axes=None,
    **kwargs,
):
    """Each bead channel against DNA, with its gate outlined and the beads picked out.

    The figure that shows what the bead gate actually took: one panel per bead
    channel, density underneath, gated events in red, the rectangle drawn on.

    Parameters
    ----------
    adata
        Mass cytometry AnnData, gated with :func:`gate_beads`.
    beads
        Bead set, as for :func:`bead_channels`.
    dna
        DNA channel, as for :func:`dna_channel`.
    bead_key
        ``obs`` column marking bead events; the highlight is skipped when it
        is missing, so a gate can be previewed before it is applied.
    gates
        Gates to outline. Defaults to the ones recorded by
        :func:`gate_beads`, falling back to the defaults.
    cofactor
        Arcsinh cofactor the gates are defined on.
    axes
        Existing axes to draw the panels on, one per bead channel.
    **kwargs
        Passed to :func:`~cytopy.plot_biaxial`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, one panel per bead channel.
    """
    import matplotlib.pyplot as plt

    from .plotting import plot_biaxial

    recorded = adata.uns.get("cytopy", {}).get("beads", {})
    if gates is None and recorded.get("gates"):
        gates = {k: {a: list(v) for a, v in g.items()} for k, g in recorded["gates"].items()}
    resolved = bead_gates(adata, beads, gates)
    dna_name = dna_channel(adata, dna)

    if axes is None:
        fig, axes = plt.subplots(
            1, len(resolved), figsize=(3.4 * len(resolved), 3.4), squeeze=False
        )
        axes = axes.ravel()
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].figure

    for axis, (name, gate) in zip(axes, resolved.items()):
        (x_lo, x_hi), (y_lo, y_hi) = gate["x"], gate["y"]
        plot_biaxial(
            adata,
            name,
            dna_name,
            cofactor=cofactor,
            color_by=bead_key if bead_key in adata.obs else "density",
            rectangles=[(x_lo, x_hi, y_lo, y_hi)],
            title=name,
            ax=axis,
            **kwargs,
        )
    fig.tight_layout()
    return fig


def _bead_curves(
    adata: ad.AnnData,
    before,
    after: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    bead_cols: Sequence[int],
    names: Sequence[str],
    k: int,
    endrule: str,
    sample_key: str | None,
    max_points: int,
) -> dict:
    """Smoothed bead intensities either side of normalisation, thinned for keeping.

    Held in ``uns`` so the figure survives the beads being removed, which is
    what a report at the end of a pipeline needs.
    """
    curves: dict[str, dict] = {}
    for label, rows in _sample_groups(adata, sample_key):
        beads_here = rows[mask[rows]]
        if beads_here.size == 0:
            continue
        beads_here = beads_here[np.argsort(times[beads_here], kind="stable")]
        step = max(int(np.ceil(beads_here.size / max_points)), 1)
        entry: dict[str, object] = {"time": times[beads_here][::step].astype(np.float32)}
        for panel, source in (("before", before), ("after", after)):
            block = np.asarray(source[np.ix_(beads_here, list(bead_cols))], dtype=np.float64)
            smoothed = np.column_stack(
                [_runmed(block[:, j], k, endrule) for j in range(len(names))]
            )
            entry[panel] = smoothed[::step].astype(np.float32)
        curves[str(label)] = entry
    return {"channels": list(names), "samples": curves}


def _plot_stashed_curves(curves: Mapping[str, object], max_points: int, ax):
    """Redraw the before/after figure from what normalisation kept in ``uns``."""
    import matplotlib.pyplot as plt

    names = [str(c) for c in curves["channels"]]
    samples = dict(curves["samples"])
    panels = ["before", "after"]
    if ax is None:
        fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3 * len(panels)), sharex=True)
        axes = np.atleast_1d(axes)
    else:
        fig, axes = ax.figure, np.atleast_1d(ax)[: len(panels)]

    colours = plt.get_cmap("tab10")
    for axis, panel in zip(axes, panels):
        for first, entry in enumerate(samples.values()):
            entry = dict(entry)
            time = np.asarray(entry["time"], dtype=np.float64)
            smoothed = np.asarray(entry[panel], dtype=np.float64)
            step = max(int(np.ceil(time.size / max_points)), 1) if max_points else 1
            for j, name in enumerate(names):
                axis.plot(
                    time[::step],
                    smoothed[::step, j],
                    color=colours(j % 10),
                    lw=1.2,
                    label=name if first == 0 else None,
                )
            for j in range(len(names)):
                axis.axhline(
                    float(np.mean(smoothed[:, j])),
                    color=colours(j % 10),
                    lw=0.7,
                    ls="--",
                    alpha=0.8,
                )
        axis.set_title(panel, loc="left", fontsize=10)
        axis.set_ylabel("smoothed intensity")
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("time")
    axes[0].legend(fontsize=8, ncol=len(names), frameon=False)
    fig.tight_layout()
    return fig
