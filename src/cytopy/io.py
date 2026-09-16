"""Reading FCS files into :class:`anndata.AnnData`."""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .spillover import _parse_spillover_keyword

__all__ = ["concat_samples", "read_fcs", "read_fcs_dir"]

# Channels that are not fluorescence measurements and should be excluded from
# transformation / compensation by default.
_SCATTER_RE = re.compile(r"^(fsc|ssc|bsc)[-_ ]?", re.IGNORECASE)
_TIME_RE = re.compile(r"^time$", re.IGNORECASE)


def _channel_kind(pnn: str) -> str:
    if _TIME_RE.match(pnn.strip()):
        return "time"
    if _SCATTER_RE.match(pnn.strip()):
        return "scatter"
    return "fluor"


def _parse_pne(value: str | None) -> tuple[float, float]:
    """Parse a ``$PnE`` amplification keyword into ``(decades, offset)``."""
    if not value:
        return (0.0, 0.0)
    parts = [p.strip() for p in str(value).split(",")]
    try:
        f1 = float(parts[0])
        f2 = float(parts[1]) if len(parts) > 1 else 0.0
    except ValueError:
        return (0.0, 0.0)
    # A zero offset with non-zero decades means "1" by the FCS3.1 spec.
    if f1 > 0 and f2 == 0:
        f2 = 1.0
    return (f1, f2)


def _linearise(x: np.ndarray, f1: float, f2: float, pnr: float) -> np.ndarray:
    """Undo log amplification stored via ``$PnE`` (mostly FCS2.0 files)."""
    if f1 <= 0:
        return x
    return f2 * np.power(10.0, f1 * x / pnr)


def _build_var(flow_data, channel_count: int) -> pd.DataFrame:
    """Pivot the flat ``$Pn*`` keywords into one row per channel.

    An FCS TEXT segment stores channel metadata as numbered keywords —
    ``$P1N``, ``$P1S``, ``$P2N``, ... — so it has to be transposed into a table
    before it can be ``adata.var``. flowio's own ``.channels`` only exposes
    ``$PnN`` and ``$PnS``; ``$PnR`` and ``$PnE`` are needed to linearise log
    amplification, and ``$PnG`` for gain.

    The ``label`` and ``kind`` columns are cytopy's, not the file's: ``label``
    is the marker-and-detector name used for the var index and the axis titles,
    ``kind`` is the scatter/fluor/time split that decides which channels
    transforms and compensation touch by default.
    """
    text = {k.lower().lstrip("$"): v for k, v in flow_data.text.items()}
    rows = []
    for i in range(1, channel_count + 1):
        pnn = str(text.get(f"p{i}n", f"P{i}")).strip()
        pns = str(text.get(f"p{i}s", "")).strip()
        try:
            pnr = float(text.get(f"p{i}r", 262144))
        except (TypeError, ValueError):
            pnr = 262144.0
        try:
            png = float(text.get(f"p{i}g", 1) or 1)
        except (TypeError, ValueError):
            png = 1.0
        f1, f2 = _parse_pne(text.get(f"p{i}e"))
        rows.append(
            {
                "channel": pnn,
                "marker": pns,
                "label": f"{pns} ({pnn})" if pns and pns != pnn else pnn,
                "kind": _channel_kind(pnn),
                "pnr": pnr,
                "png": png,
                "pne_decades": f1,
                "pne_offset": f2,
            }
        )
    var = pd.DataFrame(rows)
    # Index on the human-readable label; de-duplicate defensively.
    labels = var["label"].tolist()
    seen: dict[str, int] = {}
    unique = []
    for lab in labels:
        if lab in seen:
            seen[lab] += 1
            unique.append(f"{lab}.{seen[lab]}")
        else:
            seen[lab] = 0
            unique.append(lab)
    var.index = pd.Index(unique, name="var_names")
    return var


def _spillover_from_text(text: dict) -> pd.DataFrame | None:
    """Parse ``$SPILLOVER`` / ``$SPILL`` into a DataFrame indexed by detector."""
    lowered = {k.lower().lstrip("$"): v for k, v in text.items()}
    raw = lowered.get("spillover") or lowered.get("spill")
    if not raw:
        return None
    return _parse_spillover_keyword(raw)


def read_fcs(
    path: str | os.PathLike,
    *,
    sample_id: str | None = None,
    linearise: bool = True,
    apply_gain: bool = False,
    store_raw: bool = True,
    dtype: str = "float32",
    ignore_offset_error: bool = False,
) -> ad.AnnData:
    """Read a single FCS file into an :class:`~anndata.AnnData`.

    Parameters
    ----------
    path
        Path to the ``.fcs`` file.
    sample_id
        Name recorded in ``adata.obs['sample']``. Defaults to the file stem.
    linearise
        Undo ``$PnE`` log amplification. Older instruments stored the output of
        an analog log amplifier, where the true intensity is
        ``f2 * 10 ** (f1 * value / $PnR)``; leaving that alone would mean
        transforming already-logged values. Modern digital files write
        ``$PnE = 0,0`` and this is a no-op.
    apply_gain
        Divide linear channels by their ``$PnG`` gain. Off by default, matching
        flowCore's ``linearize``: instruments disagree about whether the gain
        has already been applied to the stored values, so applying it blindly
        silently rescales the data. Turn it on if you know your files need it.
    store_raw
        Keep an untouched copy of the values in ``adata.layers['raw']``, so
        later transforms always have something to go back to. Costs one extra
        copy of the matrix; turn it off for very large files.
    dtype
        dtype of ``adata.X`` and ``adata.layers['raw']``.
    ignore_offset_error
        Read on despite an inconsistent data-segment offset in the header.
        Some instruments write offsets that disagree with the TEXT segment;
        the file is usually still readable, so this is the escape hatch for
        one that would otherwise raise.

    Returns
    -------
    AnnData with one observation per event and one variable per channel.
    ``.X`` and ``.layers['raw']`` both start out as the values read from the
    file; ``.X`` is the working matrix, ``layers['raw']`` stays put.
    ``.var`` holds the ``$Pn*`` channel metadata, ``.uns['fcs']`` the header
    keywords and, when present, ``.uns['spillover']``.
    """
    import flowio

    path = Path(path)
    fd = flowio.FlowData(str(path), ignore_offset_error=ignore_offset_error)

    n_ch = int(fd.channel_count)
    X = np.reshape(np.asarray(fd.events, dtype=np.float64), (-1, n_ch))
    var = _build_var(fd, n_ch)

    if linearise:
        for j, (_, row) in enumerate(var.iterrows()):
            if row["pne_decades"] > 0:
                X[:, j] = _linearise(X[:, j], row["pne_decades"], row["pne_offset"], row["pnr"])
    if apply_gain:
        gains = var["png"].to_numpy(dtype=float)
        linear = var["pne_decades"].to_numpy(dtype=float) <= 0
        scale = np.where(linear & (gains > 0), gains, 1.0)
        X = X / scale

    name = sample_id or path.stem
    obs = pd.DataFrame(
        {"sample": pd.Categorical([name] * X.shape[0]), "file": [str(path)] * X.shape[0]},
        index=pd.Index([f"{name}_{i}" for i in range(X.shape[0])], name="obs_names"),
    )

    adata = ad.AnnData(X=np.ascontiguousarray(X, dtype=dtype), obs=obs, var=var)
    if store_raw:
        adata.layers["raw"] = adata.X.copy()
    adata.uns["fcs"] = {str(k): str(v) for k, v in fd.text.items()}
    spill = _spillover_from_text(fd.text)
    if spill is not None:
        adata.uns["spillover"] = spill
    timestep = adata.uns["fcs"].get("$TIMESTEP") or adata.uns["fcs"].get("timestep")
    if timestep:
        try:
            adata.uns["timestep"] = float(timestep)
        except ValueError:
            pass
    return adata


def read_fcs_dir(
    directory: str | os.PathLike,
    *,
    pattern: str = "*.fcs",
    recursive: bool = False,
    **kwargs,
) -> ad.AnnData:
    """Read every FCS file in a directory and concatenate them into one object.

    Parameters
    ----------
    directory
        Directory to search.
    pattern
        Glob matched against the file names.
    recursive
        Search sub-directories too.
    **kwargs
        Passed through to :func:`read_fcs` for every file.

    Returns
    -------
    AnnData
        All events stacked, in sorted file-name order, with
        ``adata.obs['sample']`` naming the file each event came from.

    Raises
    ------
    FileNotFoundError
        If nothing matches ``pattern``.
    """
    directory = Path(directory)
    globber = directory.rglob if recursive else directory.glob
    paths = sorted(globber(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} in {directory}")
    return concat_samples([read_fcs(p, **kwargs) for p in paths])


def concat_samples(adatas: Sequence[ad.AnnData] | Iterable[ad.AnnData]) -> ad.AnnData:
    """Stack per-sample AnnData objects on the channels they share.

    Parameters
    ----------
    adatas
        One AnnData per sample, as returned by :func:`read_fcs`. Panels need
        not match: the intersection of the channels is kept, so a channel
        missing from one file is dropped from all. A single object is returned
        untouched.

    Returns
    -------
    AnnData
        The concatenation, with unique ``obs_names`` and ``var``/``uns`` taken
        from the first object that has them.
    """
    adatas = list(adatas)
    if len(adatas) == 1:
        return adatas[0]
    with warnings.catch_warnings():
        # Two files hold the same event names often enough -- the same file
        # twice, or two runs numbered from zero. anndata warns, and the very
        # next line is what it is asking for, so the warning is just noise.
        warnings.filterwarnings("ignore", message=".*Observation names are not unique.*")
        out = ad.concat(adatas, join="inner", index_unique=None, merge="first", uns_merge="first")
    out.obs_names_make_unique()
    return out
