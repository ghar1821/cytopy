"""A record of what was removed, and why.

Every step that drops events -- a bead gate, a doublet cutoff, a debarcoding
assignment -- appends to ``adata.uns['cytopy']['filters']``. The point is that
by the end of a pipeline you can say where each event went without having kept
the intermediates around, and :func:`~cytopy.report` can render it.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

__all__ = ["filter_events", "filter_log", "record_filter"]

#: Columns of the filter log, in the order the report shows them.
_COLUMNS = ("step", "reason", "n_before", "n_removed", "pct_removed", "n_after", "params")


def record_filter(
    adata: ad.AnnData,
    step: str,
    keep: np.ndarray,
    *,
    reason: str = "",
    params: dict | None = None,
) -> dict:
    """Note that a step would keep ``keep`` and drop the rest.

    Records the counts only; nothing is removed from ``adata``. Use it when a
    step marks events rather than dropping them, so the tally still adds up.

    Parameters
    ----------
    adata
        The AnnData to annotate. Modified in place.
    step
        Short name for the step, e.g. ``"bead gate"``. Repeating a name
        appends another entry rather than replacing the first.
    keep
        Boolean mask over the events the step keeps.
    reason
        One line on why the dropped events were dropped, for the report.
    params
        The settings the step ran with, recorded alongside the counts as JSON
        so that the log survives ``write_h5ad``.

    Returns
    -------
    dict
        The entry that was appended.

    Raises
    ------
    ValueError
        If ``keep`` is not one entry per event.
    """
    keep = np.asarray(keep, dtype=bool)
    if keep.shape[0] != adata.n_obs:
        raise ValueError(f"keep has {keep.shape[0]} entries, expected {adata.n_obs}")
    n_before = int(keep.shape[0])
    n_keep = int(keep.sum())
    entry = {
        "step": str(step),
        "reason": str(reason),
        "n_before": n_before,
        "n_removed": n_before - n_keep,
        "n_after": n_keep,
        "pct_removed": 100.0 * (n_before - n_keep) / max(n_before, 1),
        # JSON, not a nested dict: the log has to survive write_h5ad, and a
        # per-entry dict of arbitrary keys does not.
        "params": json.dumps({k: _plain(v) for k, v in (params or {}).items()}, sort_keys=True),
    }
    # Held column-wise for the same reason -- anndata writes a dict of arrays,
    # but not a list of dicts.
    log = adata.uns.setdefault("cytopy", {}).setdefault("filters", {})
    for column in _COLUMNS:
        # list(), not append: a log restored from h5ad comes back as arrays.
        log[column] = [*list(log.get(column, [])), entry[column]]
    return entry


def _plain(value):
    """Anything json can write, or its repr."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    return repr(value)


def filter_events(
    adata: ad.AnnData,
    keep: np.ndarray,
    *,
    step: str,
    reason: str = "",
    params: dict | None = None,
    copy: bool = True,
) -> ad.AnnData:
    """Drop events, keeping a record of how many went and why.

    Parameters
    ----------
    adata
        The AnnData to filter.
    keep
        Boolean mask over the events to keep.
    step
        Short name for the step, recorded in the filter log.
    reason
        One line on why the dropped events were dropped.
    params
        The settings the step ran with.
    copy
        Return a new object. ``False`` still returns the subset -- AnnData
        cannot drop rows in place -- but is accepted so callers can be
        explicit that the original is not expected to survive.

    Returns
    -------
    AnnData
        The kept events, carrying the filter log forward.

    Raises
    ------
    ValueError
        If ``keep`` is not one entry per event.
    """
    record_filter(adata, step, keep, reason=reason, params=params)
    keep = np.asarray(keep, dtype=bool)
    out = adata[keep]
    return out.copy() if copy else out


def filter_log(adata: ad.AnnData) -> pd.DataFrame:
    """The filter log as a table, in the order the steps ran.

    Parameters
    ----------
    adata
        AnnData carrying ``uns['cytopy']['filters']``.

    Returns
    -------
    DataFrame
        One row per step: ``step``, ``reason``, ``n_before``, ``n_removed``,
        ``pct_removed``, ``n_after``. Empty when nothing has been recorded.
    """
    log = adata.uns.get("cytopy", {}).get("filters") or {}
    columns = [c for c in _COLUMNS if c != "params"]
    if not len(log.get("step", [])):
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame({c: list(log[c]) for c in columns}, columns=columns)
    frame["n_before"] = frame["n_before"].astype(int)
    frame["n_removed"] = frame["n_removed"].astype(int)
    frame["n_after"] = frame["n_after"].astype(int)
    return frame
