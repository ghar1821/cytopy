"""Single-cell debarcoding, following CATALYST.

Samples barcoded with a combination of metal tags are pooled, acquired
together, and separated again afterwards. The method here is CATALYST's
``assignPrelim`` / ``estCutoffs`` / ``applyCutoffs``, in three steps:

1. **Assign.** Within each event the barcode channels are ranked. For a scheme
   where every barcode has the same number of positive channels -- the usual
   doublet-filtering design -- the top *k* are called positive, which gives a
   binary code to look up in the scheme.
2. **Separate.** Barcode intensities are rescaled per population by the 95th
   percentile of their own positive channels, and each event gets a *delta*:
   the gap between its lowest positive and highest negative barcode. A
   confidently assigned cell has a large gap; a doublet has almost none.
3. **Cut.** Each population gets a separation cutoff, estimated from where its
   yield curve turns over, plus a Mahalanobis distance cutoff. Events below
   either are left unassigned.

Unassigned events keep the barcode ID ``"0"``, as they do in CATALYST.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from itertools import combinations

import anndata as ad
import numpy as np
import pandas as pd

from .transforms import channel_index

__all__ = [
    "BARCODE_SCHEMES",
    "apply_cutoffs",
    "assign_prelim",
    "barcode_key",
    "barcode_stats",
    "debarcode",
    "estimate_cutoffs",
    "plot_barcode_events",
    "plot_yields",
]

#: Named barcoding kits, as ``(masses, positive channels per barcode)``.
#: ``pd20`` is the Fluidigm/Standard BioTools Cell-ID 20-Plex Pd kit: six
#: palladium channels, three positive in each of the 20 barcodes.
BARCODE_SCHEMES: dict[str, tuple[tuple[int, ...], int]] = {
    "pd20": ((102, 104, 105, 106, 108, 110), 3),
}

#: Separation cutoffs are searched on this grid, as in CATALYST.
SEPARATIONS = np.round(np.arange(0, 1.005, 0.01), 2)

#: Barcode ID given to events that could not be assigned.
UNASSIGNED = "0"


def barcode_key(
    scheme: str | pd.DataFrame | Mapping[str, Sequence[int]],
    *,
    ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build the debarcoding scheme as a 0/1 table of barcode by channel.

    Parameters
    ----------
    scheme
        A key of :data:`BARCODE_SCHEMES` (``"pd20"``); a DataFrame indexed by
        barcode ID whose columns are the barcode masses or channel names; or a
        mapping from barcode ID to the masses that are positive in it,
        ``{"A1": [102, 104, 105], ...}``.
    ids
        Names for the barcodes, when the scheme does not carry them. Defaults
        to ``A1``, ``A2``, ... for a named kit.

    Returns
    -------
    DataFrame
        Rows are barcode IDs, columns are masses or channel names, values 0/1.

    Raises
    ------
    KeyError
        If ``scheme`` names a kit that does not exist.
    ValueError
        If the table holds anything but 0 and 1, or ``ids`` is the wrong
        length.
    """
    if isinstance(scheme, str):
        try:
            masses, k = BARCODE_SCHEMES[scheme.lower()]
        except KeyError:
            raise KeyError(
                f"unknown barcoding scheme {scheme!r}; known kits are "
                f"{sorted(BARCODE_SCHEMES)}, or pass your own table"
            ) from None
        rows = list(combinations(masses, k))
        names = list(ids) if ids is not None else [f"A{i + 1}" for i in range(len(rows))]
        if len(names) != len(rows):
            raise ValueError(f"{scheme!r} has {len(rows)} barcodes but {len(names)} ids were given")
        table = pd.DataFrame(
            [[1 if m in row else 0 for m in masses] for row in rows],
            index=pd.Index(names, name="bc_id"),
            columns=[str(m) for m in masses],
        )
        return table

    if isinstance(scheme, pd.DataFrame):
        table = scheme.copy()
    else:
        positives = {str(k): [str(m) for m in v] for k, v in scheme.items()}
        columns = sorted({m for v in positives.values() for m in v}, key=_mass_sort_key)
        table = pd.DataFrame(
            [[1 if c in v else 0 for c in columns] for v in positives.values()],
            index=pd.Index(list(positives), name="bc_id"),
            columns=columns,
        )
    if ids is not None:
        if len(ids) != table.shape[0]:
            raise ValueError(f"{table.shape[0]} barcodes but {len(ids)} ids were given")
        table.index = pd.Index([str(i) for i in ids], name="bc_id")
    table.index = pd.Index([str(i) for i in table.index], name="bc_id")
    table.columns = [str(c) for c in table.columns]
    values = table.to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise ValueError("a debarcoding scheme holds only 0 and 1")
    if UNASSIGNED in table.index:
        raise ValueError(f"{UNASSIGNED!r} is reserved for unassigned events; rename that barcode")
    return table.astype(int)


def _mass_sort_key(name: str):
    digits = re.findall(r"\d+", str(name))
    return (int(digits[0]) if digits else 0, str(name))


def _resolve_key(adata: ad.AnnData, key: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Match the scheme's columns to channels, keeping the scheme's order."""
    columns, missing = [], []
    for name in key.columns:
        try:
            columns.append(channel_index(adata, name))
        except KeyError:
            resolved = _match_mass(adata, name)
            if resolved is None:
                missing.append(name)
            else:
                columns.append(resolved)
    if missing:
        raise KeyError(
            f"no channel for barcode mass(es) {missing}; "
            f"available: {[str(c) for c in adata.var_names]}"
        )
    resolved_key = key.copy()
    resolved_key.columns = [str(adata.var_names[j]) for j in columns]
    return resolved_key, columns


def _match_mass(adata: ad.AnnData, name: str) -> int | None:
    """Find a channel by the mass number in its name, e.g. 102 -> Pd102Di."""
    digits = re.findall(r"\d+", str(name))
    if not digits:
        return None
    pattern = re.compile(rf"(?<!\d){digits[0]}(?!\d)")
    detectors = adata.var.get("channel", adata.var_names)
    for j, detector in enumerate(detectors):
        if pattern.search(str(detector)):
            return j
    return None


def _barcode_block(
    adata: ad.AnnData, columns: Sequence[int], layer: str | None, cofactor: float | None
) -> np.ndarray:
    """The barcode channels as events x channels, arcsinh-transformed."""
    matrix = adata.X if layer is None else adata.layers[layer]
    block = np.asarray(matrix[:, list(columns)], dtype=np.float64)
    if cofactor is not None:
        block = np.arcsinh(block / cofactor)
    return block


def _order(block: np.ndarray) -> np.ndarray:
    """Descending order of the barcode channels within each event."""
    return np.argsort(-block, axis=1, kind="stable")


def _ranks(order: np.ndarray) -> np.ndarray:
    """Invert an ordering: which place each channel came in, per event."""
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(order.shape[1])[None, :], axis=1)
    return ranks


def _codes_and_exclusions(
    block: np.ndarray, n_positive: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Binary barcode per event, plus events whose positives are still too low.

    ``n_positive`` is the fixed number of positive channels for a
    doublet-filtering scheme, or ``None`` when the scheme has a varying number
    and the split goes at the largest gap instead.
    """
    order = _order(block)
    n = block.shape[0]
    if n_positive is not None:
        lowest_positive = block[np.arange(n), order[:, n_positive - 1]]
        # Only this branch needs the ordering itself; inverting it costs
        # another index array the size of the barcode block.
        del order
        codes = block >= lowest_positive[:, None]
        # A large negative value can look well separated from zero; CATALYST
        # excludes events whose weakest positive is not actually positive.
        excluded = lowest_positive < 0
    else:
        ranks = _ranks(order)
        sorted_block = np.take_along_axis(block, order, axis=1)
        gaps = np.abs(np.diff(sorted_block, axis=1))
        largest = np.argmax(gaps, axis=1)
        codes = ranks <= largest[:, None]
        excluded = np.where(codes, block, np.inf).min(axis=1) < 0
    return codes, excluded


def _lookup_ids(codes: np.ndarray, key: pd.DataFrame) -> np.ndarray:
    """Map each event's binary code to a barcode ID, or to unassigned."""
    weights = (2 ** np.arange(1, codes.shape[1] + 1)).astype(np.int64)
    reference = key.to_numpy(dtype=np.int64) @ weights
    observed = codes.astype(np.int64) @ weights

    order = np.argsort(reference, kind="stable")
    sorted_reference = reference[order]
    position = np.clip(np.searchsorted(sorted_reference, observed), 0, sorted_reference.size - 1)
    matched = sorted_reference[position] == observed
    index = np.where(matched, order[position], -1)

    ids = np.asarray([UNASSIGNED, *key.index], dtype=object)
    return ids[index + 1]


def _deltas(scaled: np.ndarray, n_positive: int | None) -> np.ndarray:
    """Separation between the lowest positive and highest negative barcode."""
    n = scaled.shape[0]
    order = np.argsort(np.where(np.isnan(scaled), -np.inf, -scaled), axis=1, kind="stable")
    if n_positive is not None:
        lowest_positive = scaled[np.arange(n), order[:, n_positive - 1]]
        highest_negative = scaled[np.arange(n), order[:, n_positive]]
        return lowest_positive - highest_negative
    sorted_block = np.take_along_axis(scaled, order, axis=1)
    gaps = np.abs(np.diff(sorted_block, axis=1))
    return gaps[np.arange(n), np.nanargmax(np.where(np.isnan(gaps), -np.inf, gaps), axis=1)]


def assign_prelim(
    adata: ad.AnnData,
    key: str | pd.DataFrame | Mapping[str, Sequence[int]] = "pd20",
    *,
    layer: str | None = None,
    cofactor: float | None = 5.0,
    scaled_layer: str | None = None,
    copy: bool = False,
) -> ad.AnnData:
    """Assign a preliminary barcode to every event, and score its separation.

    Parameters
    ----------
    adata
        Mass cytometry AnnData holding raw counts in ``.X``.
    key
        Debarcoding scheme, as for :func:`barcode_key`.
    layer
        Input layer. ``None`` uses ``adata.X``.
    cofactor
        Arcsinh cofactor applied to the barcode channels before ranking them,
        matching CATALYST's use of its ``exprs`` assay. ``None`` takes the
        layer as already transformed.
    scaled_layer
        Layer to keep the per-population rescaled barcode intensities in.
        ``None`` discards them once the deltas are computed, which on a run of
        millions of events saves a whole copy of the matrix; CATALYST always
        keeps them.
    copy
        Return a modified copy instead of writing to ``adata`` in place.

    Returns
    -------
    AnnData
        The annotated object, with the barcode in ``adata.obs['bc_id']``
        (``"0"`` where nothing matched), an untouched copy of it in
        ``adata.obs['bc_prelim']`` so the yield plots stay meaningful after
        cutoffs are applied, the separation in ``adata.obs['delta']``, and the
        scheme in ``adata.uns['cytopy']['debarcode']``.

    Raises
    ------
    KeyError
        If a barcode channel cannot be matched.
    """
    adata = adata.copy() if copy else adata
    table = barcode_key(key)
    table, columns = _resolve_key(adata, table)

    row_sums = table.to_numpy().sum(axis=1)
    n_positive = int(row_sums[0]) if len(set(row_sums.tolist())) == 1 else None

    block = _barcode_block(adata, columns, layer, cofactor)
    codes, excluded = _codes_and_exclusions(block, n_positive)
    bc_id = _lookup_ids(codes, table)
    bc_id[excluded] = UNASSIGNED

    # Rescale each population by the 95th percentile of its own positive
    # channels, so that separations are comparable between barcodes.
    scaled = np.full_like(block, np.nan)
    positives = {i: np.flatnonzero(row) for i, row in zip(table.index, table.to_numpy())}
    for identifier, positive_cols in positives.items():
        rows = np.flatnonzero(bc_id == identifier)
        if rows.size == 0:
            continue
        divisor = np.quantile(block[np.ix_(rows, positive_cols)], 0.95)
        if divisor == 0:
            continue
        scaled[rows] = block[rows] / divisor
    delta = _deltas(scaled, n_positive)

    categories = [UNASSIGNED, *table.index]
    adata.obs["bc_id"] = pd.Categorical(bc_id, categories=categories)
    # Kept so the yield plots still show what each cutoff discarded once
    # apply_cutoffs has moved those events into the unassigned pile.
    adata.obs["bc_prelim"] = pd.Categorical(bc_id.copy(), categories=categories)
    adata.obs["delta"] = delta
    if scaled_layer is not None:
        full = np.zeros(adata.shape, dtype=np.float32)
        full[:, columns] = scaled.astype(np.float32)
        adata.layers[scaled_layer] = full

    info = adata.uns.setdefault("cytopy", {}).setdefault("debarcode", {})
    info.update(
        {
            "key": {str(i): [int(v) for v in row] for i, row in zip(table.index, table.to_numpy())},
            "channels": list(table.columns),
            "n_positive": n_positive if n_positive is not None else 0,
            "cofactor": float(cofactor) if cofactor is not None else 0.0,
            "layer": layer or "X",
        }
    )
    return adata


def _yields(delta: np.ndarray, bc_id: np.ndarray, ids: Sequence[str]) -> pd.DataFrame:
    """Fraction of each population surviving each separation cutoff."""
    out = np.zeros((len(SEPARATIONS), len(ids)))
    for j, identifier in enumerate(ids):
        values = delta[bc_id == identifier]
        values = values[np.isfinite(values)]
        if values.size:
            out[:, j] = (values[None, :] >= SEPARATIONS[:, None]).mean(axis=1)
    return pd.DataFrame(out, index=pd.Index(SEPARATIONS, name="cutoff"), columns=list(ids))


def _log_logistic(x: np.ndarray, b: float, d: float, e: float) -> np.ndarray:
    """drc's ``LL.3``: ``d / (1 + exp(b * (log(x) - log(e))))``, zero lower limit."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return d / (1.0 + np.exp(b * (np.log(x) - np.log(e))))


def _fit_cutoff(seps: np.ndarray, yields: np.ndarray) -> float:
    """CATALYST's estimate: a weighted blend of a linear and a log-logistic fit.

    The log-logistic term is the separation at which the yield curve first
    starts to fall away in earnest; the linear term is where a straight line
    through the curve reaches half its intercept. They are averaged by how well
    each fits, so a population whose yield curve is not sigmoidal at all still
    gets a sensible answer.
    """
    from scipy.optimize import curve_fit

    slope, intercept = np.polyfit(seps, yields, 1)
    linear_estimate = -intercept / (2.0 * slope) if slope != 0 else np.nan
    rss_linear = float(np.sum((yields - (slope * seps + intercept)) ** 2))

    positive = seps > 0
    try:
        half = np.interp(0.5 * yields[0], yields[positive][::-1], seps[positive][::-1])
        guess = (5.0, max(yields[0], 1e-6), max(half, 1e-3))
        params, _ = curve_fit(
            _log_logistic,
            seps[positive],
            yields[positive],
            p0=guess,
            maxfev=10_000,
        )
    except (RuntimeError, TypeError, ValueError):
        return float(linear_estimate)

    b, d, e = params
    fitted = _log_logistic(seps, b, d, e)
    fitted[~np.isfinite(fitted)] = d
    rss_log = float(np.sum((yields - fitted) ** 2))

    # First cutoff at which the curve's relative slope exceeds 1%.
    with np.errstate(divide="ignore", invalid="ignore"):
        power = np.exp(b * (np.log(seps) - np.log(e)))
        derivative = -d * power * b / seps / (1.0 + power) ** 2
    derivative[0] = 0.0
    relative = np.abs(derivative) / np.where(fitted == 0, np.nan, fitted)
    hits = np.flatnonzero(np.isfinite(relative) & (relative > 1e-2))
    log_estimate = float(seps[hits[0]]) if hits.size else np.nan

    if not np.isfinite(log_estimate):
        return float(linear_estimate)
    if not np.isfinite(linear_estimate):
        return log_estimate
    total = rss_log + rss_linear
    weight = rss_log / total if total > 0 else 0.5
    return float(weight * linear_estimate + (1.0 - weight) * log_estimate)


def estimate_cutoffs(adata: ad.AnnData) -> pd.Series:
    """Estimate a separation cutoff for each barcode population.

    Parameters
    ----------
    adata
        AnnData that has been through :func:`assign_prelim`.

    Returns
    -------
    Series
        Cutoff per barcode ID, also stored in
        ``adata.uns['cytopy']['debarcode']['sep_cutoffs']``. ``NaN`` where no
        curve could be fitted, which :func:`apply_cutoffs` treats as 1, i.e.
        it discards the population rather than guessing.

    Raises
    ------
    KeyError
        If :func:`assign_prelim` has not been run.
    """
    info = _debarcode_info(adata)
    ids = list(info["key"])
    column = "bc_prelim" if "bc_prelim" in adata.obs else "bc_id"
    yields = _yields(
        np.asarray(adata.obs["delta"], dtype=np.float64),
        np.asarray(adata.obs[column], dtype=object),
        ids,
    )
    estimates = {
        identifier: _fit_cutoff(SEPARATIONS, yields[identifier].to_numpy()) for identifier in ids
    }
    info["sep_cutoffs"] = {k: float(v) for k, v in estimates.items()}
    return pd.Series(estimates, name="sep_cutoff")


def _debarcode_info(adata: ad.AnnData) -> dict:
    info = adata.uns.get("cytopy", {}).get("debarcode")
    if not info or "bc_id" not in adata.obs:
        raise KeyError("run cytopy.assign_prelim first")
    return info


def apply_cutoffs(
    adata: ad.AnnData,
    *,
    sep_cutoffs: float | Mapping[str, float] | pd.Series | None = None,
    mhl_cutoff: float = 30.0,
    layer: str | None = None,
    cofactor: float | None = None,
    copy: bool = False,
) -> ad.AnnData:
    """Unassign the events that are not confidently barcoded.

    Two rules, both CATALYST's: an event whose separation falls below its
    population's cutoff is unassigned, and so is one lying more than
    ``mhl_cutoff`` squared Mahalanobis units from the centre of its population
    in barcode space -- which is what removes the doublets that landed on a
    real barcode by accident.

    Parameters
    ----------
    adata
        AnnData that has been through :func:`assign_prelim`.
    sep_cutoffs
        One cutoff for every population, or a mapping from barcode ID to its
        own. ``None`` uses the estimates from :func:`estimate_cutoffs`, and
        treats a population without one as unusable.
    mhl_cutoff
        Squared Mahalanobis distance beyond which an event is unassigned.
        CATALYST's default is 30; raise it to keep more events.
    layer
        Layer the distances are measured on. Defaults to what
        :func:`assign_prelim` used.
    cofactor
        Arcsinh cofactor for that layer. Defaults to what
        :func:`assign_prelim` used.
    copy
        Return a modified copy instead of writing to ``adata`` in place.

    Returns
    -------
    AnnData
        The annotated object, with ``obs['bc_id']`` set to ``"0"`` for every
        event that failed either rule, the squared distances in
        ``obs['mhl_dist']``, and the cutoffs used recorded in ``uns``.

    Raises
    ------
    KeyError
        If :func:`assign_prelim` has not been run, or a cutoff names an
        unknown barcode.
    """
    from .filters import record_filter

    adata = adata.copy() if copy else adata
    info = _debarcode_info(adata)
    ids = list(info["key"])
    cutoffs = _resolve_cutoffs(info, ids, sep_cutoffs)

    if layer is None:
        stored = str(info.get("layer", "X"))
        layer = None if stored == "X" else stored
    if cofactor is None:
        cofactor = info.get("cofactor") or None
    columns = [channel_index(adata, c) for c in info["channels"]]
    block = _barcode_block(adata, columns, layer, cofactor)

    bc_id = np.asarray(adata.obs["bc_id"], dtype=object)
    delta = np.asarray(adata.obs["delta"], dtype=np.float64)
    before = bc_id != UNASSIGNED

    distance = np.full(adata.n_obs, np.nan)
    drop_separation = np.zeros(adata.n_obs, dtype=bool)
    for identifier in ids:
        rows = np.flatnonzero(bc_id == identifier)
        if rows.size == 0:
            continue
        fails = delta[rows] < cutoffs[identifier]
        drop_separation[rows[fails]] = True
        kept = rows[~fails]
        if kept.size <= len(columns):
            continue
        values = block[kept]
        covariance = np.cov(values, rowvar=False)
        try:
            precision = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            continue
        centred = values - values.mean(axis=0)
        distance[kept] = np.einsum("ij,jk,ik->i", centred, precision, centred)

    drop_distance = np.nan_to_num(distance, nan=-np.inf) > mhl_cutoff
    bc_id[drop_separation | drop_distance] = UNASSIGNED

    adata.obs["bc_id"] = pd.Categorical(bc_id, categories=[UNASSIGNED, *ids])
    adata.obs["mhl_dist"] = distance
    info["sep_cutoffs_used"] = {k: float(v) for k, v in cutoffs.items()}
    info["mhl_cutoff"] = float(mhl_cutoff)

    record_filter(
        adata,
        "debarcode",
        bc_id != UNASSIGNED,
        reason=(
            f"{int((drop_separation & before).sum()):,} below their separation cutoff, "
            f"{int((drop_distance & before).sum()):,} beyond a Mahalanobis distance of "
            f"{mhl_cutoff:g}, {int((~before).sum()):,} matched no barcode"
        ),
        params={"mhl_cutoff": float(mhl_cutoff)},
    )
    return adata


def _resolve_cutoffs(info: dict, ids: Sequence[str], sep_cutoffs) -> dict[str, float]:
    if sep_cutoffs is None:
        estimates = info.get("sep_cutoffs")
        if estimates is None:
            raise KeyError("no separation cutoffs; run cytopy.estimate_cutoffs or pass them")
        # A population with no estimate is discarded rather than guessed at.
        return {
            i: (1.0 if not np.isfinite(estimates.get(i, np.nan)) else float(estimates[i]))
            for i in ids
        }
    if isinstance(sep_cutoffs, int | float):
        return dict.fromkeys(ids, float(sep_cutoffs))
    given = dict(sep_cutoffs)
    unknown = sorted(set(given) - set(ids))
    if unknown:
        raise KeyError(f"cutoffs given for unknown barcode(s) {unknown}")
    return {i: float(given.get(i, 1.0)) for i in ids}


def debarcode(
    adata: ad.AnnData,
    key: str | pd.DataFrame | Mapping[str, Sequence[int]] = "pd20",
    *,
    layer: str | None = None,
    cofactor: float | None = 5.0,
    sep_cutoffs: float | Mapping[str, float] | pd.Series | None = None,
    mhl_cutoff: float = 30.0,
    copy: bool = False,
) -> ad.AnnData:
    """Assign barcodes, estimate the cutoffs, and apply them.

    The three steps run separately when you want to look at the yield plots
    before committing to the cutoffs; this is the shortcut for when you do not.

    Parameters
    ----------
    adata
        Mass cytometry AnnData holding raw counts in ``.X``.
    key
        Debarcoding scheme, as for :func:`barcode_key`.
    layer
        Input layer. ``None`` uses ``adata.X``.
    cofactor
        Arcsinh cofactor for the barcode channels.
    sep_cutoffs
        Separation cutoffs. ``None`` estimates them.
    mhl_cutoff
        Squared Mahalanobis cutoff.
    copy
        Return a modified copy instead of writing to ``adata`` in place.

    Returns
    -------
    AnnData
        The annotated object, debarcoded.
    """
    adata = adata.copy() if copy else adata
    assign_prelim(adata, key, layer=layer, cofactor=cofactor)
    if sep_cutoffs is None:
        estimate_cutoffs(adata)
    return apply_cutoffs(adata, sep_cutoffs=sep_cutoffs, mhl_cutoff=mhl_cutoff)


def barcode_stats(adata: ad.AnnData) -> pd.DataFrame:
    """Events per barcode, with the cutoffs that produced them.

    Parameters
    ----------
    adata
        AnnData that has been through :func:`assign_prelim`.

    Returns
    -------
    DataFrame
        One row per barcode plus ``"0"`` for the unassigned: ``n``, ``pct``,
        ``sep_cutoffs`` and the barcode's positive channels.

    Raises
    ------
    KeyError
        If :func:`assign_prelim` has not been run.
    """
    info = _debarcode_info(adata)
    ids = list(info["key"])
    cutoffs = info.get("sep_cutoffs_used") or info.get("sep_cutoffs") or {}
    counts = adata.obs["bc_id"].value_counts()
    channels = list(info["channels"])
    rows = []
    for identifier in [UNASSIGNED, *ids]:
        n = int(counts.get(identifier, 0))
        pattern = info["key"].get(identifier)
        rows.append(
            {
                "bc_id": identifier,
                "n": n,
                "pct": 100.0 * n / max(adata.n_obs, 1),
                "sep_cutoff": float(cutoffs.get(identifier, np.nan)),
                "channels": ""
                if pattern is None
                else "+".join(c for c, v in zip(channels, pattern) if v),
            }
        )
    return pd.DataFrame(rows).set_index("bc_id")


def plot_yields(
    adata: ad.AnnData,
    which: str | Sequence[str] | None = None,
    *,
    ncols: int = 4,
    axes=None,
):
    """Yield against separation cutoff, per barcode -- CATALYST's yield plot.

    Grey bars are how many events sit at each separation; the red line is the
    fraction of the population that survives a cutoff there; the dashed line is
    the cutoff in force. A good population has a shoulder: most events well
    separated, and a cutoff placed just before the curve falls off a cliff.

    Parameters
    ----------
    adata
        AnnData that has been through :func:`assign_prelim`.
    which
        Barcodes to draw. ``None`` draws them all, one panel each; pass a
        single ID for a single panel, or ``"all"`` for one panel with every
        population's curve overlaid.
    ncols
        Panels per row.
    axes
        Existing axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Raises
    ------
    KeyError
        If :func:`assign_prelim` has not been run, or ``which`` names an
        unknown barcode.
    """
    import matplotlib.pyplot as plt

    info = _debarcode_info(adata)
    ids = list(info["key"])
    cutoffs = info.get("sep_cutoffs_used") or info.get("sep_cutoffs") or {}
    delta = np.asarray(adata.obs["delta"], dtype=np.float64)
    # Against the preliminary assignment, so the curve shows the population as
    # it was before the cutoff carved it up.
    column = "bc_prelim" if "bc_prelim" in adata.obs else "bc_id"
    bc_id = np.asarray(adata.obs[column], dtype=object)
    yields = _yields(delta, bc_id, ids)

    overlay = isinstance(which, str) and which == "all"
    if overlay or which is None:
        selected = ids
    else:
        selected = [which] if isinstance(which, str) else list(which)
        unknown = sorted(set(selected) - set(ids))
        if unknown:
            raise KeyError(f"unknown barcode(s) {unknown}; the scheme has {ids}")

    width = SEPARATIONS[1] - SEPARATIONS[0]
    if overlay:
        if axes is None:
            fig, axis = plt.subplots(figsize=(7, 3.5))
        else:
            axis = np.atleast_1d(axes)[0]
            fig = axis.figure
        counts = np.histogram(delta[np.isfinite(delta)], bins=np.r_[SEPARATIONS, 1.01])[0]
        axis.bar(SEPARATIONS, counts, width=width, color="lightgrey", edgecolor="white", lw=0.2)
        scale = max(counts.max(), 1)
        colours = plt.get_cmap("Spectral")
        for j, identifier in enumerate(ids):
            axis.plot(
                SEPARATIONS,
                yields[identifier].to_numpy() * scale,
                color=colours(j / max(len(ids) - 1, 1)),
                lw=1.0,
            )
        _style_yield_axis(axis, scale)
        axis.set_title(f"all barcodes ({len(ids)})", fontsize=9, loc="left")
        fig.tight_layout()
        return fig

    nrows = int(np.ceil(len(selected) / ncols))
    if axes is None:
        fig, grid = plt.subplots(
            nrows,
            min(ncols, len(selected)),
            figsize=(3.4 * min(ncols, len(selected)), 2.6 * nrows),
            squeeze=False,
        )
        grid = grid.ravel()
    else:
        grid = np.atleast_1d(axes)
        fig = grid[0].figure

    for axis, identifier in zip(grid, selected):
        values = delta[(bc_id == identifier) & np.isfinite(delta)]
        counts = np.histogram(values, bins=np.r_[SEPARATIONS, 1.01])[0]
        scale = max(counts.max(), 1)
        axis.bar(SEPARATIONS, counts, width=width, color="darkgrey", edgecolor="white", lw=0.2)
        axis.plot(SEPARATIONS, yields[identifier].to_numpy() * scale, color="red", lw=1.2)
        cutoff = cutoffs.get(identifier, np.nan)
        if np.isfinite(cutoff):
            axis.axvline(cutoff, color="blue", ls="--", lw=1.0)
            survived = float(np.mean(values >= cutoff) * 100) if values.size else 0.0
            axis.annotate(
                f"cutoff {cutoff:.2f}\nyield {survived:.1f}%",
                xy=(cutoff, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(4 if cutoff < 0.55 else -4, -4),
                textcoords="offset points",
                fontsize=7,
                color="blue",
                va="top",
                ha="left" if cutoff < 0.55 else "right",
            )
        _style_yield_axis(axis, scale)
        pattern = "+".join(c for c, v in zip(info["channels"], info["key"][identifier]) if v)
        axis.set_title(f"{identifier}: {pattern}  ({values.size:,})", fontsize=8, loc="left")
    for axis in grid[len(selected) :]:
        axis.set_visible(False)
    fig.tight_layout()
    return fig


def _style_yield_axis(axis, scale: float) -> None:
    axis.set_xlim(-0.025, 1.025)
    axis.set_ylim(0, scale * 1.05)
    axis.set_xlabel("barcode separation", fontsize=8)
    axis.set_ylabel("event count", fontsize=8)
    axis.tick_params(labelsize=7)
    secondary = axis.twinx()
    secondary.set_ylim(0, 105)
    secondary.set_ylabel("yield", fontsize=8)
    secondary.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    secondary.tick_params(labelsize=7)
    axis.spines[["top"]].set_visible(False)
    secondary.spines[["top"]].set_visible(False)


def plot_barcode_events(
    adata: ad.AnnData,
    which: str | Sequence[str] | None = None,
    *,
    n: int = 100,
    layer: str | None = None,
    cofactor: float | None = None,
    ncols: int = 4,
    seed: int = 0,
    axes=None,
):
    """Barcode intensities of a sample of events, per population.

    CATALYST's event plot: one column of dots per event, one colour per
    barcode channel. A clean population separates into a positive band and a
    negative band with nothing in between; a population that does not is one
    to look at before trusting it.

    Parameters
    ----------
    adata
        AnnData that has been through :func:`assign_prelim`.
    which
        Barcodes to draw, including ``"0"`` for the unassigned. ``None`` draws
        every population that has events.
    n
        Events sampled per population.
    layer
        Layer to read. Defaults to what :func:`assign_prelim` used.
    cofactor
        Arcsinh cofactor. Defaults to what :func:`assign_prelim` used.
    ncols
        Panels per row.
    seed
        Seed for the event sample, so a report is reproducible.
    axes
        Existing axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Raises
    ------
    KeyError
        If :func:`assign_prelim` has not been run, or ``which`` names an
        unknown barcode.
    """
    import matplotlib.pyplot as plt

    info = _debarcode_info(adata)
    ids = list(info["key"])
    bc_id = np.asarray(adata.obs["bc_id"], dtype=object)

    if which is None:
        selected = [i for i in [UNASSIGNED, *ids] if np.any(bc_id == i)]
    else:
        selected = [which] if isinstance(which, str) else list(which)
        unknown = sorted(set(selected) - {UNASSIGNED, *ids})
        if unknown:
            raise KeyError(f"unknown barcode(s) {unknown}; the scheme has {ids}")

    if layer is None:
        stored = str(info.get("layer", "X"))
        layer = None if stored == "X" else stored
    if cofactor is None:
        cofactor = info.get("cofactor") or None
    columns = [channel_index(adata, c) for c in info["channels"]]
    channels = list(info["channels"])

    nrows = int(np.ceil(len(selected) / ncols))
    if axes is None:
        fig, grid = plt.subplots(
            nrows,
            min(ncols, len(selected)),
            figsize=(3.4 * min(ncols, len(selected)), 2.6 * nrows),
            squeeze=False,
        )
        grid = grid.ravel()
    else:
        grid = np.atleast_1d(axes)
        fig = grid[0].figure

    rng = np.random.default_rng(seed)
    colours = plt.get_cmap("Spectral")
    matrix = adata.X if layer is None else adata.layers[layer]
    for axis, identifier in zip(grid, selected):
        rows = np.flatnonzero(bc_id == identifier)
        total = rows.size
        if total > n:
            rows = np.sort(rng.choice(rows, n, replace=False))
        block = np.asarray(matrix[np.ix_(rows, columns)], dtype=np.float64)
        if cofactor is not None:
            block = np.arcsinh(block / cofactor)
        for j, channel in enumerate(channels):
            axis.scatter(
                np.arange(block.shape[0]),
                block[:, j],
                s=4,
                color=colours(j / max(len(channels) - 1, 1)),
                linewidths=0,
                label=channel if axis is grid[0] else None,
            )
        label = "unassigned" if identifier == UNASSIGNED else identifier
        axis.set_title(f"{label} ({total:,} events)", fontsize=8, loc="left")
        axis.set_xticks([])
        axis.set_ylabel("intensity", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in grid[len(selected) :]:
        axis.set_visible(False)
    fig.tight_layout()
    # Outside the panels: the positive band sits where a legend would go.
    handles, labels = grid[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=7,
        ncol=min(len(labels), 8),
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        markerscale=2,
    )
    return fig
