"""Command line entry point: ``cytopy sample.fcs``."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Read the file named on the command line, transform it, and open the viewer.

    Parameters
    ----------
    argv
        Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status; ``0`` once the window has been closed.
    """
    p = argparse.ArgumentParser(prog="cytopy", description=__doc__)
    p.add_argument(
        "path",
        type=Path,
        nargs="+",
        help="FCS file, directory of FCS files, or .h5ad; several become samples",
    )
    p.add_argument("-x", "--x-channel", default=None)
    p.add_argument("-y", "--y-channel", default=None)
    p.add_argument("--ticks", default="auto", choices=["auto", "linear"])
    p.add_argument("--cofactor", type=float, default=150.0)
    p.add_argument("--asinh", action="store_true", help="arcsinh transform, then plot that layer")
    p.add_argument("--logicle", action="store_true", help="logicle transform, then plot that layer")
    p.add_argument("--compensate", action="store_true", help="apply the $SPILLOVER matrix")
    p.add_argument("--spillover", type=Path, default=None, help="compensate with a matrix CSV")
    p.add_argument(
        "--controls",
        type=Path,
        default=None,
        help="directory of single-stain controls; derive the matrix from them and compensate",
    )
    p.add_argument(
        "--normalise",
        nargs="?",
        const="fluidigm",
        default=None,
        metavar="BEADS",
        help="bead-normalise before plotting; optionally name the bead set "
        "(fluidigm/dvs, beta, xt). Uses the default gate -- check it in the beads panel.",
    )
    p.add_argument("--subsample", type=int, default=0, help="plot at most N events")
    p.add_argument("--bins", type=int, default=512)
    args = p.parse_args(argv)

    import cytopy
    from cytopy.viewer import as_one_anndata

    adata = as_one_anndata(args.path if len(args.path) > 1 else args.path[0])
    print(adata)

    layer = None
    if sum(map(bool, [args.compensate, args.spillover, args.controls])) > 1:
        p.error("pick one of --compensate / --spillover / --controls")
    if args.controls:
        controls, unstained = cytopy.read_controls(args.controls)
        spillover = cytopy.spillover_from_controls(controls, unstained=unstained)
        print(spillover.round(4))
    else:
        spillover = args.spillover
    if args.compensate or args.spillover or args.controls:
        cytopy.compensate(adata, spillover)
        layer = "comp"
    if args.normalise:
        cytopy.normalise_beads(adata, beads=args.normalise)
        info = adata.uns["cytopy"]["beads"]
        print(f"beads: {info['samples']}")
        layer = "normalised"
    if args.asinh and args.logicle:
        p.error("pick one of --asinh / --logicle")
    if args.asinh:
        cytopy.asinh_transform(adata, args.cofactor, layer=layer)
        layer = "asinh"
    if args.logicle:
        cytopy.logicle_transform(adata, layer=layer)
        layer = "logicle"
    if args.subsample:
        adata = cytopy.subsample(adata, args.subsample)

    cytopy.view(
        adata,
        layer=layer,
        x=args.x_channel,
        y=args.y_channel,
        ticks=args.ticks,
        bins=args.bins,
        block=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
