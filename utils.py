import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ovrlpy._plotting import BIH_CMAP as cmap

CM = 1 / 2.54


def finalize_axes(ax, x, y, window_size):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if x is not np.nan:
        ax.set_xlim(x - window_size, x + window_size)
    if y is not np.nan:
        ax.set_ylim(y - window_size, y + window_size)

    if y is not np.nan:
        ax.invert_yaxis()
    ax.invert_xaxis()


def plot_transcripts(
    roi_transcripts,
    signal_integrity,
    signal_strength,
    x,
    y,
    window_size,
    figsize=(24 * CM, 5 * CM),
):
    fig = plt.figure(figsize=figsize)

    gs = plt.GridSpec(2, 4)

    ax_top = plt.subplot(gs[:, 0])
    ax_top.scatter(
        x=roi_transcripts["x"],
        y=roi_transcripts["y"],
        c=roi_transcripts["RGB"],
        s=1,
        # rasterized=True,
    )
    finalize_axes(ax_top, x, y, window_size)
    ax_top.set_title("top")

    ax_bottom = plt.subplot(gs[:, 1])
    ax_bottom.scatter(
        x=roi_transcripts["x"].reverse(),
        y=roi_transcripts["y"].reverse(),
        c=roi_transcripts["RGB"].reverse(),
        s=1,
        # rasterized=True,
    )
    finalize_axes(ax_bottom, x, y, window_size)
    ax_bottom.set_title("bottom")

    ax_horizontal_x = plt.subplot(gs[0, 2])
    roi_x = roi_transcripts.filter(pl.col("y") < (y + 4), pl.col("y") > (y - 4))
    ax_horizontal_x.scatter(
        x=roi_x["x"],
        y=roi_x["z"],
        c=roi_x["RGB"],
        s=1,
        # rasterized=True,
    )
    finalize_axes(ax_horizontal_x, x, np.nan, window_size)
    ax_horizontal_x.set_title("side (x)")

    ax_horizontal_y = plt.subplot(gs[1, 2])
    roi_y = roi_transcripts.filter(pl.col("x") < (x + 4), pl.col("x") > (x - 4))
    ax_horizontal_y.scatter(
        x=roi_y["y"],
        y=roi_y["z"],
        c=roi_y["RGB"],
        s=1,
        # rasterized=True,
    )
    finalize_axes(ax_horizontal_y, y, np.nan, window_size)
    ax_horizontal_y.set_title("side (y)")

    ax_integrity = plt.subplot(gs[:, 3], facecolor="black")
    ax_integrity.imshow(
        signal_integrity,
        alpha=(signal_strength / 3).clip(0, 1).astype(float),
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    finalize_axes(ax_integrity, window_size, window_size, window_size)
    ax_integrity.set_title("vertical signal integrity")

    return fig


def plot_doublet(ovrlp, x, y, window_size=50, figsize=(24 * CM, 5 * CM)):
    signal_integrity = ovrlp.integrity_map[
        int(y) - window_size : int(y) + window_size,
        int(x) - window_size : int(x) + window_size,
    ]

    signal_strength = ovrlp.signal_map[
        int(y) - window_size : int(y) + window_size,
        int(x) - window_size : int(x) + window_size,
    ]

    roi_transcripts = ovrlp.subset_transcripts(x, y, window_size=window_size).sort("z")
    _, embedding_color = ovrlp.transform_transcripts(roi_transcripts)
    roi_transcripts = roi_transcripts.with_columns(RGB=embedding_color)

    return plot_transcripts(
        roi_transcripts, signal_integrity, signal_strength, x, y, window_size, figsize
    )
