"""Preprocessing of cytometry AnnData objects: compensation and transforms."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from .scales import LogicleScale

__all__ = [
    "asinh_transform",
    "channel_index",
    "compensate",
    "estimate_cofactors",
    "fluor_channels",
    "logicle_transform",
    "subsample",
]

DEFAULT_COFACTOR = {"flow": 150.0, "cytof": 5.0, "spectral": 3000.0}


def channel_index(adata: ad.AnnData, name: str) -> int:
    """Resolve a channel name to its column index.

    Parameters
    ----------
    adata
        Cytometry AnnData.
    name
        A ``var_name``, a ``$PnN`` detector (``"FITC-A"``), a ``$PnS`` marker
        (``"CD3"``) or the combined label (``"CD3 (FITC-A)"``). Matched in that
        order, then case-insensitively against the var_names.

    Returns
    -------
    int
        Column index into ``adata.X`` and every layer.

    Raises
    ------
    KeyError
        If nothing matches. The message lists the available channels.
    """
    if name in adata.var_names:
        return int(adata.var_names.get_loc(name))
    for col in ("channel", "marker", "label"):
        if col in adata.var:
            hits = np.flatnonzero(adata.var[col].astype(str).values == name)
            if hits.size:
                return int(hits[0])
    lowered = {str(v).lower(): i for i, v in enumerate(adata.var_names)}
    if name.lower() in lowered:
        return lowered[name.lower()]
    raise KeyError(f"channel {name!r} not found; available: {list(adata.var_names)}")


def fluor_channels(adata: ad.AnnData) -> list[str]:
    """The fluorescence channels, i.e. everything but scatter and time.

    This is the default set that compensation and the transforms act on.

    Parameters
    ----------
    adata
        Cytometry AnnData. Without a ``var['kind']`` column (which
        :func:`~cytopy.read_fcs` adds) every channel is returned.

    Returns
    -------
    list of str
        var_names, in file order.
    """
    if "kind" not in adata.var:
        return list(adata.var_names)
    return list(adata.var_names[adata.var["kind"].values == "fluor"])


def _resolve_channels(adata: ad.AnnData, channels: Sequence[str] | None) -> list[int]:
    if channels is None:
        names = fluor_channels(adata)
    else:
        names = list(channels)
    return [channel_index(adata, n) for n in names]


def _get_matrix(adata: ad.AnnData, layer: str | None) -> np.ndarray:
    X = adata.X if layer is None else adata.layers[layer]
    return np.asarray(X, dtype=np.float64)


# --------------------------------------------------------------------------
# compensation
# --------------------------------------------------------------------------
def compensate(
    adata: ad.AnnData,
    spillover: pd.DataFrame | np.ndarray | str | os.PathLike | None = None,
    *,
    layer: str | None = None,
    key_added: str = "comp",
    copy: bool = False,
) -> ad.AnnData:
    """Apply the spillover matrix, undoing fluorescence crosstalk between detectors.

    Compensated values are ``X @ inv(S)``.

    Parameters
    ----------
    adata
        Cytometry AnnData.
    spillover
        Square spillover matrix, in any of the forms cytopy can get one:

        * ``None`` (default) uses ``adata.uns['spillover']``, which
          :func:`~cytopy.read_fcs` parses from the file's ``$SPILLOVER``;
        * a DataFrame, whose index/columns name the detectors and need only
          cover a subset of the channels — as returned by
          :func:`~cytopy.spillover_from_controls`, which derives one from
          single-stain controls;
        * a path to a CSV exported by other software, read with
          :func:`~cytopy.read_spillover`;
        * a bare array, assumed to be in :func:`fluor_channels` order.
    layer
        Input layer. ``None`` uses ``adata.X``.
    key_added
        Layer to write the compensated matrix to.
    copy
        Return a modified copy instead of writing to ``adata`` in place.

    Returns
    -------
    AnnData
        The annotated object, with the compensated matrix in
        ``adata.layers[key_added]`` and where the matrix came from in
        ``adata.uns['cytopy']['spillover_source']``. Channels absent from the
        matrix are copied through untouched.

    Raises
    ------
    ValueError
        If no matrix is given and the file recorded none, if the matrix is not
        square, if its rows and columns do not name the same detectors, or if
        it cannot be inverted.
    """
    adata = adata.copy() if copy else adata
    source = "argument"
    if spillover is None:
        spillover = adata.uns.get("spillover")
        source = "uns"
    if spillover is None:
        raise ValueError("no spillover matrix given and none found in adata.uns['spillover']")
    if isinstance(spillover, str | os.PathLike):
        from .spillover import read_spillover

        source = f"csv:{spillover}"
        spillover = read_spillover(spillover)
    if not isinstance(spillover, pd.DataFrame):
        names = fluor_channels(adata)
        values = np.asarray(spillover, dtype=float)
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError(f"spillover must be a square matrix, got shape {values.shape}")
        if values.shape[0] != len(names):
            raise ValueError(
                f"spillover is {values.shape[0]}x{values.shape[1]} but there are "
                f"{len(names)} fluorescence channels; pass a DataFrame to name them"
            )
        spillover = pd.DataFrame(values, index=names, columns=names)
    if spillover.shape[0] != spillover.shape[1]:
        raise ValueError(f"spillover must be square, got shape {spillover.shape}")

    # Rows and columns must describe the same detectors, but need not list them
    # in the same order (or under the same alias), so resolve both and align.
    cols = [channel_index(adata, c) for c in spillover.columns]
    rows = [channel_index(adata, r) for r in spillover.index]
    if sorted(rows) != sorted(cols):
        raise ValueError(
            f"spillover rows {list(spillover.index)} and columns {list(spillover.columns)} "
            "do not name the same detectors"
        )
    if rows != cols:
        spillover = spillover.iloc[[rows.index(c) for c in cols]]

    X = _get_matrix(adata, layer)
    out = X.copy()
    try:
        inv = np.linalg.inv(spillover.to_numpy(dtype=float))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"spillover matrix is singular and cannot be inverted: {exc}") from exc
    out[:, cols] = X[:, cols] @ inv
    adata.layers[key_added] = out.astype(adata.X.dtype, copy=False)
    cytopy_uns = adata.uns.setdefault("cytopy", {})
    cytopy_uns["compensated_layer"] = key_added
    cytopy_uns["spillover_source"] = source
    return adata


# --------------------------------------------------------------------------
# arcsinh
# --------------------------------------------------------------------------
def asinh_transform(
    adata: ad.AnnData,
    cofactor: float | Mapping[str, float] = 150.0,
    *,
    channels: Sequence[str] | None = None,
    layer: str | None = None,
    key_added: str | None = "asinh",
    copy: bool = False,
) -> ad.AnnData:
    """Arcsinh-transform channels: ``asinh(x / cofactor)``.

    Parameters
    ----------
    adata
        Cytometry AnnData (events x channels).
    cofactor
        A single cofactor for every channel, or a mapping from channel name to
        cofactor. Typical values: 5 for mass cytometry, 150 for flow,
        ~3000 for spectral flow. Channels missing from a mapping keep the
        mapping's ``"default"`` entry, or 150.
    channels
        Channels to transform. Defaults to the fluorescence channels; scatter
        and time channels are copied through unchanged.
    layer
        Input layer. ``None`` uses ``adata.X``. Pass ``"comp"`` to transform
        compensated data.
    key_added
        Layer to write. ``None`` overwrites ``adata.X`` instead.
    copy
        Return a modified copy instead of writing to ``adata`` in place.

    Returns
    -------
    AnnData
        The annotated object, with the transformed matrix in
        ``adata.layers[key_added]`` and the per-channel cofactors in
        ``adata.var['cofactor']`` (``NaN`` for untransformed channels). The
        viewer reads both back to label its axes in raw units.
    """
    adata = adata.copy() if copy else adata
    idx = _resolve_channels(adata, channels)

    if isinstance(cofactor, Mapping):
        default = float(cofactor.get("default", 150.0))
        cofs = np.full(adata.n_vars, np.nan)
        for j in idx:
            name = str(adata.var_names[j])
            chan = str(adata.var["channel"].iloc[j]) if "channel" in adata.var else name
            mark = str(adata.var["marker"].iloc[j]) if "marker" in adata.var else name
            for key in (name, mark, chan):
                if key in cofactor:
                    cofs[j] = float(cofactor[key])
                    break
            else:
                cofs[j] = default
    else:
        cofs = np.full(adata.n_vars, np.nan)
        cofs[idx] = float(cofactor)

    X = _get_matrix(adata, layer)
    out = X.copy()
    sel = np.asarray(idx, dtype=int)
    out[:, sel] = np.arcsinh(X[:, sel] / cofs[sel][None, :])

    out = out.astype(np.float32, copy=False)
    if key_added is None:
        adata.X = out
    else:
        adata.layers[key_added] = out
    adata.var["cofactor"] = cofs
    info = adata.uns.setdefault("cytopy", {})
    info["asinh_layer"] = key_added
    # Per layer as well, so a second transform does not orphan the first.
    info.setdefault("asinh_layers", {})[str(key_added)] = {
        str(n): float(c) for n, c in zip(adata.var_names, cofs) if np.isfinite(c)
    }
    return adata


def estimate_cofactors(
    adata: ad.AnnData,
    *,
    channels: Sequence[str] | None = None,
    layer: str | None = None,
    quantile: float = 0.05,
    minimum: float = 1.0,
) -> dict[str, float]:
    """Rough per-channel cofactors from the spread of the negative population.

    Uses ``|quantile|`` of the negative values, which puts the linear region of
    the arcsinh roughly where the noise is. Meant as a starting point to be
    eyeballed in the viewer, not as a substitute for choosing them by hand.

    Parameters
    ----------
    adata
        Cytometry AnnData.
    channels
        Channels to estimate for. Defaults to :func:`fluor_channels`.
    layer
        Input layer. ``None`` uses ``adata.X``; pass ``"comp"`` to estimate
        from compensated data, which is usually what you want.
    quantile
        Quantile of the negative values to take as the noise width. Larger
        values give larger cofactors and a wider linear region.
    minimum
        Floor on the returned cofactors, so a channel with no negatives cannot
        produce a zero.

    Returns
    -------
    dict
        Maps var_name to cofactor, ready to pass to :func:`asinh_transform`.
    """
    idx = _resolve_channels(adata, channels)
    X = _get_matrix(adata, layer)
    out: dict[str, float] = {}
    for j in idx:
        col = X[:, j]
        col = col[np.isfinite(col)]
        neg = col[col < 0]
        if neg.size >= 50:
            cof = abs(float(np.quantile(neg, quantile)))
        else:
            pos = col[col > 0]
            cof = float(np.quantile(pos, 0.05)) if pos.size else minimum
        out[str(adata.var_names[j])] = max(cof, minimum)
    return out


# --------------------------------------------------------------------------
# logicle
# --------------------------------------------------------------------------
def logicle_transform(
    adata: ad.AnnData,
    *,
    channels: Sequence[str] | None = None,
    layer: str | None = None,
    key_added: str | None = "logicle",
    T: float | None = None,
    M: float = 4.5,
    W: float | None = None,
    A: float = 0.0,
    copy: bool = False,
) -> ad.AnnData:
    """Logicle (biexponential) transform, per channel, onto a 0..1 display scale.

    Parameters
    ----------
    adata
        Cytometry AnnData.
    channels
        Channels to transform. Defaults to :func:`fluor_channels`; scatter and
        time channels are copied through unchanged.
    layer
        Input layer. ``None`` uses ``adata.X``. Pass ``"comp"`` to transform
        compensated data.
    key_added
        Layer to write. ``None`` overwrites ``adata.X`` instead.
    T
        Top of scale: the data value that maps to 1.0. Defaults per channel to
        that channel's maximum.
    M
        Total number of decades the scale covers.
    W
        Decades of linearisation around zero. Defaults per channel to a fit
        against the spread of that channel's negative values; see
        :meth:`~cytopy.scales.LogicleScale.from_data`.
    A
        Additional decades of negative data shown below zero.
    copy
        Return a modified copy instead of writing to ``adata`` in place.

    Returns
    -------
    AnnData
        The annotated object, with the transformed matrix in
        ``adata.layers[key_added]`` and the per-channel ``T``/``W``/``M``/``A``
        in ``adata.uns['cytopy']['logicle_params']``, which the viewer reads
        back to label its axes in raw units.
    """
    adata = adata.copy() if copy else adata
    idx = _resolve_channels(adata, channels)
    X = _get_matrix(adata, layer)
    out = X.copy()
    params: dict[str, dict[str, float]] = {}
    for j in idx:
        col = X[:, j]
        if W is None:
            scale = LogicleScale.from_data(col, T=T, M=M, A=A)
        else:
            top = T if T is not None else float(max(np.max(col), 10.0))
            scale = LogicleScale(T=top, W=W, M=M, A=A)
        out[:, j] = scale.forward(col)
        params[str(adata.var_names[j])] = {"T": scale.T, "W": scale.W, "M": scale.M, "A": scale.A}

    out = out.astype(np.float32, copy=False)
    if key_added is None:
        adata.X = out
    else:
        adata.layers[key_added] = out
    info = adata.uns.setdefault("cytopy", {})
    info["logicle_params"] = params
    info["logicle_layer"] = key_added
    # Also by layer: transforming twice -- raw and compensated, say -- used to
    # leave the first layer unlabelled, because there was only ever one slot.
    info.setdefault("logicle_layers", {})[str(key_added)] = params
    return adata


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def subsample(
    adata: ad.AnnData,
    n: int = 200_000,
    *,
    per_sample: bool = False,
    seed: int = 0,
) -> ad.AnnData:
    """Take a random subset of events, for responsive interactive plotting.

    Parameters
    ----------
    adata
        Cytometry AnnData.
    n
        Number of events to keep — in total, or from each sample when
        ``per_sample`` is set.
    per_sample
        Sample ``n`` events from each value of ``adata.obs['sample']``, so a
        small file is not swamped by a large one. Samples with fewer than ``n``
        events are kept whole.
    seed
        Seed for the random draw, so a subset is reproducible.

    Returns
    -------
    AnnData
        A copy holding the selected events in their original order. Returned
        unchanged if it is already small enough and ``per_sample`` is off.
    """
    rng = np.random.default_rng(seed)
    if adata.n_obs <= n and not per_sample:
        return adata
    if per_sample and "sample" in adata.obs:
        keep: list[np.ndarray] = []
        for rows in adata.obs.groupby("sample", observed=True).indices.values():
            rows = np.asarray(rows)
            take = rows if rows.size <= n else rng.choice(rows, n, replace=False)
            keep.append(take)
        sel = np.sort(np.concatenate(keep))
    else:
        sel = np.sort(rng.choice(adata.n_obs, n, replace=False))
    return adata[sel].copy()
