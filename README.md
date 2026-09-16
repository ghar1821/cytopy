# cytopy

Cytometry analysis on [AnnData](https://anndata.readthedocs.io), gated
interactively in [napari](https://napari.org).

Read FCS files into an `AnnData` (events × channels), compensate and transform
them, then open a napari window that draws a biexponential density plot of any
two channels and turns polygons you draw into boolean gates in `adata.obs`.

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[gui,dev]"
```

## Use

```python
import cytopy

adata = cytopy.read_fcs("sample.fcs")        # events x channels, + layers["raw"]
cytopy.compensate(adata)                     # -> adata.layers["comp"]
cytopy.asinh_transform(adata, cofactor=150,  # -> adata.layers["asinh"]
                       layer="comp")

cytopy.view(adata, layer="asinh", x="CD3", y="CD19")
```

or from a terminal:

```bash
cytopy sample.fcs --compensate --asinh --cofactor 150 -x CD3 -y CD19
cytopy run.fcs --normalise --asinh --cofactor 5        # mass cytometry
```

### Compensation

`compensate` applies a spillover matrix `S` as `X @ inv(S)`, writing
`layers["comp"]`. There are three ways to get `S`, and all of them end up as the
same square DataFrame indexed by detector, with `1` on the diagonal:

```python
cytopy.compensate(adata)                       # 1. the file's own $SPILLOVER
cytopy.compensate(adata, "matrix.csv")         # 2. exported by other software

controls, unstained = cytopy.read_controls("controls/")   # 3. single-stain controls
spill = cytopy.spillover_from_controls(controls, unstained=unstained)
cytopy.compensate(adata, spill)
```

**From single-stain controls.** `spillover_from_controls` implements the
Bagwell and Adams matrix-inversion method. For the control stained with dye
*i*, it takes the difference between the positive and negative populations in
every detector *j*, then divides that row through by the difference in the
dye's own detector — so the coefficients do not depend on how bright the
control happened to be, and the diagonal is `1` by construction. Inverting the
assembled matrix is the N-parameter generalisation that lets every detector be
corrected at once.

The positive population is found by splitting the control's stained detector on
an arcsinh scale (Otsu), or you can hand it one: `positive_gate="P1"` uses a
boolean `obs` column, so a gate drawn in the viewer works, and `thresholds=`
takes a raw cutoff per control. The negative reference is the control's own
negative events by default — same beads, same autofluorescence — or pass
`unstained=` to use a universal negative instead. A control that does not split
into two populations is an error rather than a quietly wrong row.

`read_controls` matches each file in a directory to the detector it stains by
name (`FITC-A.fcs`, `Compensation Controls_PE-A.fcs`, `CD3 FITC.fcs` all work,
against both `$PnN` and `$PnS`), falling back to whichever channel separates
best. It picks the unstained tube out by name too.

**From a CSV.** `read_spillover` copes with what the common exporters write:
with or without a row-name column, comma / tab / semicolon separated, fractions
or percentages, and names decorated as `Comp-FITC-A` or `FITC-A :: CD3`. Pass
`inverted=True` if the file holds a compensation matrix rather than a spillover
matrix. `write_spillover` writes one back out.

From the terminal:

```bash
cytopy sample.fcs --controls controls/ --asinh -x CD3 -y CD19
cytopy sample.fcs --spillover matrix.csv --asinh
```

### Bead normalisation (mass cytometry)

Mass cytometry sensitivity drifts over an acquisition. Spiked-in metal beads do
not change, so their drift *is* the instrument's, and every mass channel can be
scaled back onto a fixed baseline — Finck et al. (2013).

```python
adata = cytopy.read_fcs("run.fcs")

cytopy.gate_beads(adata)                 # -> adata.obs["bead"]
cytopy.normalise_beads(adata)            # -> adata.layers["normalised"]
cytopy.plot_beads_over_time(adata)       # the before/after diagnostic
```

or interactively, which is the point of the **beads** panel in the viewer:
pick a bead channel to plot it against DNA, drag a rectangle round the bead
population, hit *gate from drawn rectangle*, then *apply bead gate* and
*normalise*. The gate is premessa's — one rectangle per bead channel, on
`asinh(x / 5)` — and an event is a bead only when it is inside all of them.
The default is `x = (2, 5)`, `y = (-1, 2)`; it is a starting point, not an
answer, which is why it is adjustable.

```python
cytopy.gate_beads(adata, gates={"Ce140Di": {"x": (2.5, 4.8)}})   # move one edge
cytopy.gate_beads(adata, beads="beta")                           # or "fluidigm"/"dvs", "xt"
cytopy.gate_beads(adata, beads=[140, 151, 153, 165, 175])        # or your own masses
```

Bead channels are matched by isotope against `$PnN`, so `Ce140Di`, `140Ce` and a
bare `140` all resolve; a bead set missing a channel is an error, not a warning.
`obs["bead"]` is read if it already exists, so a polygon drawn in the viewer or
any other mask works in place of the rectangles. The gate you settled on is
kept in `adata.uns["cytopy"]["beads"]["gates"]` and survives `write_h5ad`, so
you can re-apply it to the next file: `gate_beads(other, gates=saved)`.

To put several acquisitions on one scale, compute the baseline once and pass it
in; each sample is smoothed and interpolated on its own clock, since `Time`
restarts with every file.

```python
baseline = cytopy.bead_baseline(reference)
cytopy.normalise_beads(adata, baseline=baseline)
```

After normalising, `cytopy.bead_distance(adata, layer="normalised")` gives
premessa's `beadDist` — the Mahalanobis distance from the bead centroid — which
catches the bead-cell doublets a rectangular gate lets through. `remove_beads`
drops the beads and, given `distance_cutoff=`, those doublets too, recording
what went in the filter log:

```python
clean = cytopy.remove_beads(adata, distance_cutoff=5, layer="normalised")
```

**Fidelity.** The arithmetic is CATALYST's `normCytof` and premessa's
`normalize_folder`, which agree with each other: bead intensities are smoothed
over time with a running median, the correction factor is the slope of the
no-intercept least-squares fit `sum(b * B) / sum(b * b)`, and those slopes are
interpolated linearly across all events. Checked against both R implementations
run on the same data — the gate selects the identical events and the output
agrees to 1e-12 in float64. Where the two differ, the defaults here are
premessa's:

| | premessa (default) | CATALYST |
| --- | --- | --- |
| baseline | median of bead events | mean (`statistic="mean"`) |
| smoothing window | 201 events | 500 (`k=500`) |
| running-median ends | Tukey end rule | repeat (`endrule="constant"`) |
| doublets | `bead_distance` after | median±5·MAD (`trim=5`) |
| gate | rectangles you adjust | automatic |

### Debarcoding

Samples barcoded with a combination of metal tags, pooled and acquired
together, are separated again with CATALYST's method:

```python
cytopy.assign_prelim(adata, "pd20")   # -> obs["bc_id"], obs["delta"]
cytopy.estimate_cutoffs(adata)        # -> uns[...]["sep_cutoffs"]
cytopy.plot_yields(adata)             # look before you commit
cytopy.apply_cutoffs(adata, mhl_cutoff=30)

cytopy.debarcode(adata, "pd20")       # all three, when you do not want to look
```

`"pd20"` is the Fluidigm Cell-ID 20-Plex Pd kit — six palladium channels,
three positive in each of 20 barcodes. Any other scheme works as a table or a
mapping:

```python
cytopy.barcode_key({"donor1": [102, 104, 105], "donor2": [102, 104, 106]})
```

Within each event the barcode channels are ranked and the top *k* called
positive, which gives a binary code to look up. Intensities are then rescaled
per population by the 95th percentile of that population's own positive
channels, and each event scores a **delta**: the gap between its lowest
positive and its highest negative barcode. A confidently barcoded cell has a
large gap; a doublet carrying two barcodes has almost none. Each population
gets a separation cutoff estimated from where its yield curve turns over, and
a Mahalanobis cutoff on top of that. Everything failing either is left as
`"0"`, unassigned — and `obs["bc_prelim"]` keeps the call before the cutoffs,
so the yield plots still show what each one discarded.

**Fidelity.** Checked against CATALYST's `assignPrelim` / `estCutoffs` /
`applyCutoffs` run verbatim in R on the same 60,000 events: the preliminary
assignments are identical event for event, the deltas agree to 8e-16, the
Mahalanobis distances to 9e-15, and the final assignments are identical with
zero disagreements. Cutoff estimates are the one place they part company —
`drc::drm` fits the log-logistic curve and scipy does not fit it identically,
so 19 of 20 agreed to 1e-5 and one landed on the adjacent point of the 0.01
grid the estimate is searched on. It changed no assignment.

### Reports

Every step that drops events records what it dropped, so at the end you can
account for all of them without having kept the intermediates:

```python
cytopy.filter_log(adata)
#           step                                          reason  n_before  n_removed  n_after
#   remove beads  inside the bead gate (4,109 events); 0 more...     60000       4109    55891
#      debarcode  2,958 below their separation cutoff, 140 be...     55891       3098    52793
```

`cytopy.report` turns that, and the plots behind it, into one self-contained
HTML file — no assets alongside it, opens anywhere:

```python
cytopy.report(adata, "qc.html", title="Run 1", beads=before_removal)
```

It picks up whatever the data carries: the event tally at each step, the bead
before/after lines, the bead gate with the gated events picked out, every gate
in the plane it was drawn in, and the debarcoding yield and event plots. Pass
`beads=` the object from before the beads were removed if you want those
panels; the line plot needs nothing, because normalisation stashes the curves
it would need.

### Biaxial plots

`plot_biaxial` is the static counterpart to the viewer — the same smoothed 2-D
histogram, so a figure in a report matches what was on screen when the gate was
drawn. **Colour is by event density by default**: at a few million events a
per-point scatter is neither fast nor readable, and the structure is where
events pile up.

```python
cytopy.plot_biaxial(adata, "CD3", "CD19", layer="asinh")      # density
cytopy.plot_biaxial(adata, "CD3", "CD19", color_by="bead")    # a group over a grey density
cytopy.plot_biaxial(adata, "Ce140Di", "Ir193Di", cofactor=5)  # display-only arcsinh
cytopy.plot_gate(adata, "lymphocytes")                        # redraw a gate where it was drawn
```

Axes are labelled in raw units whenever the layer records the transform that
produced it, exactly as in the viewer. `cofactor=` compresses the axis for
display without writing a layer, which matters when the matrix is large.

### Large runs

Everything on the mass cytometry path is sized for runs of millions of events.
Measured on 5,000,000 events (a 2020 M1 laptop, single core):

| | 12 channels | 46 channels |
| --- | --- | --- |
| `gate_beads` | 0.26 s, +35 MB | 0.35 s, +36 MB |
| `normalise_beads` | 1.3 s, +64 MB | 3.1 s, +936 MB |
| `normalise_beads(key_added=None)` | — | 2.1 s, **+44 MB** |
| `plot_beads_over_time` | 0.32 s | — |
| viewer redraw | 0.05 s | — |
| dragging a bead gate edge | 0.02 s | — |

The memory figures are on top of the matrix itself (0.86 GB at 46 channels).
Writing a new layer costs one more copy of it, which is unavoidable;
`key_added=None` normalises `adata.X` in place and costs nothing.

What that took, in case you hit the same walls elsewhere:

* **Never widen the matrix.** Only the bead channels at the bead events are
  needed to fit the slopes — a few hundred thousand rows however big the file
  is — and the output is written a column at a time at the storage dtype. An
  intermediate `float64` copy of everything was 1.2 GB at 12 channels and over
  4 GB at 46.
* **Group by categorical codes, not strings.** `obs["sample"].astype(str)`
  builds a Python string per event; that one line was 0.8 s of the 1.4 s the
  diagnostic plot used to take.
* **Bin by arithmetic.** `density_image` computes bin indices directly and
  counts with `bincount` rather than calling `np.histogram2d`. Identical
  output, 6× faster, and it runs on every widget change.
* **Draw fewer points than you compute.** `plot_beads_over_time` smooths over
  every bead event and then thins the finished curve to `max_points` (10,000
  by default). The line is the same; matplotlib just gets less of it.
* **Estimate while dragging, count exactly on apply.** The bead panel's live
  count comes from a fixed 250,000-event subsample and is labelled `est.`;
  *apply bead gate* uses every event.

### Transforms are always explicit

The viewer never transforms anything. It plots the layer you point it at,
exactly as stored, so the plot shows the result of the transform *you* ran.
What it does infer is how to **label** the axes: `asinh_transform` and
`logicle_transform` record their parameters on the AnnData, and the viewer uses
them to put the ticks at decades of the original units — which is what turns an
arcsinh layer into a plot that reads as biexponential.

So `layers["asinh"]` gets an axis marked `-10² 0 10² 10³ 10⁴`, while plotting
raw `X` gets plain linear ticks. Set **axis ticks** to `linear` to see the
stored numbers instead.

### Several files at once

`view` takes one object or many. Several are concatenated on the channels they
share and become entries in the **sample** selector, so one window browses the
lot:

```python
cytopy.view([run1, run2, run3])                     # AnnData you already have
cytopy.view({"healthy": run1, "treated": run2})     # name them yourself
cytopy.view("data/")                                # every FCS in a directory
cytopy.view(["a.fcs", "b.h5ad"])                    # paths work too
```

Two files called the same thing stay two entries (`demo`, `demo.1`) rather than
merging. **same axes across samples** is on whenever there is more than one:
without it the axes rescale each time you switch, which is what makes two
samples look alike when they are not. Turn it off to let each sample fill the
plot.

Panels need not match — the intersection of the channels is kept. `uns` comes
from the first object, so per-file provenance (bead gates, spillover) beyond
the first is not carried over; normalise and gate before combining if you need
it kept.

### In the viewer

* Pick the two channels and the matrix to plot (`X` or any layer).
* The plot is a smoothed 2-D histogram; `log counts` is on by default so rare
  populations stay visible.
* Select the **gates** layer, draw a polygon (or rectangle/ellipse), name it and
  hit *apply gate from shapes*. The gate lands in `adata.obs[name]` as a
  boolean column and **stays on screen**, outlined on an *applied gates* layer
  and labelled with its name and its share of its own parent. The editable
  layer clears, so the next gate is only what you draw for it — draw several
  shapes before applying to make one gate out of their union.
* To change a gate later, pick it under **edit gate** and hit *load gate onto
  canvas*: the plot switches to the plane it was drawn in, the outline comes
  back editable, and applying it again replaces it. **Its children are
  recomputed**, so a hierarchy stays consistent when you move a parent. *delete
  gate* removes it and everything nested inside it.
* Outside the viewer, `cytopy.gate_mask(adata, name)` recomputes a gate from
  the outline it was drawn with, and `cytopy.recompute_gates(adata, name)`
  brings its descendants back in line.
* The **gates** layer is always kept on top — a gate you cannot see is a gate
  you cannot adjust.
* Shapes are kept in data coordinates, so changing channel, sample, parent gate
  or bin count moves the axes without moving what a shape covers.
* Clipping the axes means a few events are drawn nowhere and cannot be gated.
  Applying a gate says how many, and `cv.off_axis_count()` reports it at any
  time.
* Set **parent gate** to an existing gate to plot only those events; gates
  applied then intersect with the parent, so hierarchies compose.
* One plot. **plot** switches it between the two-channel **density** and a 1-D
  **histogram** — one smoothed distribution of the x channel per sample, each
  in its own colour with a translucent fill under it, and a legend. The y axis
  is **% of mode** — each curve scaled so its tallest point is 100, as flow
  software does — so a rare population reads against a common one. On a
  histogram a drawn shape gates the interval it spans, which is how a threshold
  is set.
* **samples** picks which samples the plot shows, in either mode. Choosing none
  draws them all; a density pools whatever is picked, a histogram draws a curve
  for each.
* The heading above the plot is `<sample> — <parent gate>`, so it always says
  what you are looking at and what it came from.
* The axes always clip to the 0.1–99.9th percentile. Without it a single
  extreme event — and compensation makes those — stretches the axis until
  everything else is a dot in the corner. Pass `robust=False` to `view` if you
  really want the full range.
* Duplicating the plot's layer in napari does not give a second plot: that
  layer is rewritten on every redraw.

### Exporting a gating hierarchy

```python
cytopy.gating_pdf(adata, "gating.pdf")
```

A contents page with the tree — every gate, its count, its share of its parent
and of the file — then one plot per gate: the biaxial it was actually drawn on,
its parent's events underneath, its own outline over them and the events it
kept picked out. Gates come out parents first, so the pages read the way the
gating was done.

Gate provenance (channels, layer, parent, counts) is kept in
`adata.uns["cytopy"]["gates"]`.

## Data model

| where | what |
| --- | --- |
| `adata.X` | the working matrix, one row per event; starts as the file's values |
| `adata.layers["raw"]` | untouched copy of what was read from the file |
| `adata.var` | `$PnN` channel, `$PnS` marker, range, gain, kind (scatter/fluor/time) |
| `adata.layers["comp"]` | compensated |
| `adata.layers["asinh"]` | arcsinh transformed |
| `adata.obs["sample"]` | source file, for concatenated runs |
| `adata.obs[<gate>]` | boolean gate membership |
| `adata.obs["bead"]` | bead events, from the bead gate |
| `adata.obs["bead_slope"]` | per-event bead correction factor |
| `adata.layers["normalised"]` | bead-normalised counts |
| `adata.obs["bc_id"]` | barcode, `"0"` for unassigned |
| `adata.obs["delta"]` | barcode separation per event |
| `adata.uns["cytopy"]["filters"]` | what each step removed, and why |
| `adata.uns["fcs"]` | the raw FCS TEXT keywords |
| `adata.uns["spillover"]` | `$SPILLOVER` as a DataFrame |
| `adata.uns["cytopy"]["spillover_source"]` | where the applied matrix came from |

Everything round-trips through `adata.write_h5ad(...)` except `uns["spillover"]`
being restored as a plain array.

Reading a file undoes `$PnE` log amplification (analog log amps on older
instruments store already-logged values; modern files write `$PnE = 0,0` and
this does nothing). `$PnG` gain is **not** applied unless you pass
`apply_gain=True` — instruments disagree about whether the gain is already
baked into the stored values, matching flowCore's `linearize` default.

## Scales

`cytopy.scales` implements the logicle / biexponential transform of
Moore & Parks (2012) with the standard `T`, `W`, `M`, `A` parameters, plus
arcsinh, log and linear. Each scale maps data to display coordinates, back
again, and knows where its decade ticks go. `PretransformedScale` is the one
the viewer uses: identity on the values, but borrowing an inner transform to
place raw-unit ticks.

`LogicleScale.from_data` picks `T` and `W` the way FlowJo and flowCore do —
`W` widens until the linear region covers the spread of the negative
population.


```bash
python examples/make_demo_fcs.py demo.fcs           # synthetic 6-channel data
python examples/make_demo_fcs.py demo.fcs controls  # + single-stain controls
python examples/make_demo_cytof.py demo_cytof.fcs    # synthetic CyTOF with drift
python examples/quickstart.py demo.fcs
pytest                                              # the viewer tests need a display
```
