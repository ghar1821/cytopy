"""Polygon gating on top of the napari canvas."""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd

__all__ = [
    "add_gate",
    "gate_children",
    "gate_mask",
    "gate_stats",
    "polygon_mask",
    "recompute_gates",
    "rectangle_to_polygon",
]


def rectangle_to_polygon(verts: np.ndarray) -> np.ndarray:
    """Normalise a napari rectangle into four polygon corners.

    napari already stores a rectangle as its four corners, so this is usually
    a pass-through; anything else is reduced to its axis-aligned bounding box.

    Parameters
    ----------
    verts
        ``(n, 2)`` array of shape vertices, in pixel coordinates.

    Returns
    -------
    ndarray
        ``(4, 2)`` array of corners, counter-clockwise from the minimum corner.
    """
    verts = np.asarray(verts, dtype=float)
    if verts.shape[0] == 4:
        return verts
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    return np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]], [hi[0], lo[1]]])


def ellipse_mask(points: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Select points inside the ellipse inscribed in a bounding box.

    Parameters
    ----------
    points
        ``(n, 2)`` array of point coordinates.
    verts
        ``(4, 2)`` corners of the ellipse's bounding box, as napari stores an
        ellipse shape. A zero-width axis never matches.

    Returns
    -------
    ndarray
        Boolean mask of length ``n``.
    """
    verts = np.asarray(verts, dtype=float)
    centre = verts.mean(axis=0)
    radii = (verts.max(axis=0) - verts.min(axis=0)) / 2.0
    radii = np.where(radii == 0, np.inf, radii)
    d = (points - centre) / radii
    return np.sum(d**2, axis=1) <= 1.0


def polygon_mask(points: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Select points falling inside a polygon.

    Parameters
    ----------
    points
        ``(n, 2)`` array of point coordinates.
    verts
        ``(m, 2)`` polygon vertices, in the same coordinate space as
        ``points``. Fewer than three vertices enclose nothing.

    Returns
    -------
    ndarray
        Boolean mask of length ``n``.
    """
    from matplotlib.path import Path

    points = np.asarray(points, dtype=float)
    verts = np.asarray(verts, dtype=float)
    if verts.shape[0] < 3:
        return np.zeros(points.shape[0], dtype=bool)
    return Path(verts).contains_points(points)


def shapes_mask(
    points: np.ndarray,
    shapes: Sequence[np.ndarray],
    shape_types: Sequence[str] | None = None,
) -> np.ndarray:
    """Select points inside any of a napari Shapes layer's shapes.

    Drawing several shapes therefore gives their union, which is how a
    population split across two blobs gets gated in one go.

    Parameters
    ----------
    points
        ``(n, 2)`` array of point coordinates, in canvas pixel space.
    shapes
        The Shapes layer's ``.data``: one vertex array per shape.
    shape_types
        The layer's ``.shape_type``, parallel to ``shapes``. ``ellipse`` is
        treated as an inscribed ellipse and everything else as a polygon.
        Defaults to treating every shape as a polygon.

    Returns
    -------
    ndarray
        Boolean mask of length ``n``.
    """
    mask = np.zeros(points.shape[0], dtype=bool)
    types = list(shape_types) if shape_types is not None else ["polygon"] * len(shapes)
    for verts, kind in zip(shapes, types):
        verts = np.asarray(verts, dtype=float)
        if kind == "ellipse":
            mask |= ellipse_mask(points, verts)
        elif kind in ("rectangle", "polygon", "path", "line"):
            mask |= polygon_mask(
                points, rectangle_to_polygon(verts) if kind == "rectangle" else verts
            )
    return mask


def add_gate(
    adata: ad.AnnData,
    name: str,
    mask: np.ndarray,
    *,
    parent: str | None = None,
    meta: dict | None = None,
) -> np.ndarray:
    """Record a gate as a boolean column in ``adata.obs``.

    Parameters
    ----------
    adata
        The AnnData to annotate. Modified in place.
    name
        Column name to write in ``adata.obs``. An existing column of the same
        name is overwritten.
    mask
        Boolean mask over all ``adata.n_obs`` events.
    parent
        Name of an existing gate to nest this one inside. The mask is
        intersected with it, so gates compose into a hierarchy the way they do
        in FlowJo. ``None`` gates the whole file.
    meta
        Extra provenance (plotted channels, layer, ...) merged into the gate's
        record in ``adata.uns['cytopy']['gates'][name]``.

    Returns
    -------
    ndarray
        The mask actually stored, i.e. after intersection with ``parent``.

    Raises
    ------
    ValueError
        If ``mask`` is not one entry per event.
    KeyError
        If ``parent`` names a gate that does not exist.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != adata.n_obs:
        raise ValueError(f"mask has {mask.shape[0]} entries, expected {adata.n_obs}")
    if parent:
        if parent not in adata.obs:
            raise KeyError(f"parent gate {parent!r} not in adata.obs")
        mask = mask & adata.obs[parent].to_numpy(dtype=bool)
    adata.obs[name] = pd.Series(mask, index=adata.obs_names)
    gates = adata.uns.setdefault("cytopy", {}).setdefault("gates", {})
    gates[name] = {"parent": parent or "", "n": int(mask.sum()), **(meta or {})}
    return mask


def gate_stats(adata: ad.AnnData, name: str, parent: str | None = None) -> dict:
    """Event count and frequency for a gate.

    Parameters
    ----------
    adata
        AnnData carrying the gate as a boolean ``obs`` column.
    name
        The gate to summarise.
    parent
        Gate to express the frequency relative to, in addition to the whole
        file. Omitted from the result when not given.

    Returns
    -------
    dict
        ``gate``, ``n``, ``pct_total`` and, with ``parent``, ``pct_parent``.
    """
    mask = adata.obs[name].to_numpy(dtype=bool)
    n = int(mask.sum())
    out = {"gate": name, "n": n, "pct_total": 100.0 * n / max(adata.n_obs, 1)}
    if parent and parent in adata.obs:
        n_parent = int(adata.obs[parent].to_numpy(dtype=bool).sum())
        out["pct_parent"] = 100.0 * n / max(n_parent, 1)
    return out


def gate_mask(adata: ad.AnnData, name: str) -> np.ndarray:
    """Recompute a recorded gate from the outline it was drawn with.

    The outline is kept in data coordinates, so this does not need the canvas
    it was drawn on. The canvas maps data to pixels with an affine transform,
    which leaves the inside of a shape unchanged, so this reproduces exactly
    what the viewer computed at the time.

    Parameters
    ----------
    adata
        AnnData carrying the gate in ``uns['cytopy']['gates']``.
    name
        The gate to recompute.

    Returns
    -------
    ndarray
        Boolean mask over all events, already intersected with the gate's
        parent.

    Raises
    ------
    KeyError
        If the gate was never recorded, or recorded no outline -- one applied
        from a mask rather than drawn cannot be recomputed.
    """
    gates = adata.uns.get("cytopy", {}).get("gates", {})
    if name not in gates:
        raise KeyError(f"no gate {name!r}; recorded gates are {sorted(gates)}")
    record = gates[name]
    vertices = record.get("vertices")
    if not vertices or "x" not in record or "y" not in record:
        raise KeyError(f"gate {name!r} has no outline to recompute from")

    from .transforms import channel_index

    layer = str(record.get("layer", "X"))
    matrix = adata.X if layer in ("X", "") else adata.layers[layer]
    xi = channel_index(adata, record["x"])
    yi = channel_index(adata, record["y"])
    points = np.column_stack(
        [
            np.asarray(matrix[:, xi], dtype=np.float64).ravel(),
            np.asarray(matrix[:, yi], dtype=np.float64).ravel(),
        ]
    )
    kinds = record.get("shape_types") or ["polygon"] * len(vertices)
    mask = shapes_mask(points, [np.asarray(v, dtype=float) for v in vertices], list(kinds))

    parent = record.get("parent") or ""
    if parent and parent in adata.obs:
        mask = mask & adata.obs[parent].to_numpy(dtype=bool)
    return mask


def gate_children(adata: ad.AnnData, name: str) -> list[str]:
    """Gates nested directly inside ``name``.

    Parameters
    ----------
    adata
        AnnData carrying the gates.
    name
        The parent gate.

    Returns
    -------
    list of str
        Child gate names, in the order they were recorded.
    """
    gates = adata.uns.get("cytopy", {}).get("gates", {})
    return [g for g, record in gates.items() if str(record.get("parent") or "") == name]


def recompute_gates(adata: ad.AnnData, name: str) -> list[str]:
    """Bring a gate's descendants back in line after it has been changed.

    A child is stored as the events inside its own outline *and* inside its
    parent, worked out when it was drawn. Move the parent and the child is
    stale until it is recomputed, which is what this does, depth first.

    Parameters
    ----------
    adata
        AnnData carrying the gates. Modified in place.
    name
        The gate that changed. It is not itself recomputed.

    Returns
    -------
    list of str
        The descendants that were updated, parents before children. A child
        with no stored outline is skipped and left as it was.
    """
    updated: list[str] = []
    for child in gate_children(adata, name):
        try:
            mask = gate_mask(adata, child)
        except KeyError:
            continue
        adata.obs[child] = pd.Series(mask, index=adata.obs_names)
        adata.uns["cytopy"]["gates"][child]["n"] = int(mask.sum())
        updated.append(child)
        updated += recompute_gates(adata, child)
    return updated
