import numpy as np
from ovrlpy._plotting import _plot_signal_integrity

CM = 1 / 2.54
FONT = {"family": "sans-serif", "weight": "normal", "size": 6}

save_kwargs = dict(dpi=600, bbox_inches="tight")
label_kwargs = dict(fontsize=14, fontweight="bold", va="top", ha="right")


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
    signal_threshold=3,
    figsize=(24 * CM, 5 * CM),
    scatter_kwargs=dict(s=1, lw=0, marker="."),
):
    import matplotlib.pyplot as plt
    import polars as pl

    fig = plt.figure(figsize=figsize)

    gs = plt.GridSpec(2, 4)

    ax_top = plt.subplot(gs[:, 0])
    roi_top = roi_transcripts.filter(pl.col("z") > pl.col("z_center"))
    ax_top.scatter(x=roi_top["x"], y=roi_top["y"], c=roi_top["RGB"], **scatter_kwargs)
    finalize_axes(ax_top, x, y, window_size)
    ax_top.set_title("top")

    ax_bottom = plt.subplot(gs[:, 1])
    roi_bottom = roi_transcripts.filter(pl.col("z") < pl.col("z_center")).reverse()
    ax_bottom.scatter(
        x=roi_bottom["x"], y=roi_bottom["y"], c=roi_bottom["RGB"], **scatter_kwargs
    )
    finalize_axes(ax_bottom, x, y, window_size)
    ax_bottom.set_title("bottom")

    ax_horizontal_x = plt.subplot(gs[0, 2])
    roi_x = roi_transcripts.filter(pl.col("y") < (y + 4), pl.col("y") > (y - 4))
    ax_horizontal_x.scatter(
        x=roi_x["x"], y=roi_x["z"], c=roi_x["RGB"], **scatter_kwargs
    )
    finalize_axes(ax_horizontal_x, x, np.nan, window_size)
    ax_horizontal_x.set_title("side (x)")

    ax_horizontal_y = plt.subplot(gs[1, 2])
    roi_y = roi_transcripts.filter(pl.col("x") < (x + 4), pl.col("x") > (x - 4))
    ax_horizontal_y.scatter(
        x=roi_y["y"], y=roi_y["z"], c=roi_y["RGB"], **scatter_kwargs
    )
    finalize_axes(ax_horizontal_y, y, np.nan, window_size)
    ax_horizontal_y.set_title("side (y)")

    ax_integrity = plt.subplot(gs[:, 3], facecolor="black")
    _plot_signal_integrity(
        ax_integrity, signal_integrity, signal_strength, signal_threshold
    )

    finalize_axes(ax_integrity, window_size, window_size, window_size)
    ax_integrity.set_title("VSI")

    return fig


def plot_doublet(
    ovrlp, x, y, window_size=50, signal_threshold=3, figsize=(24 * CM, 5 * CM)
):
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
        roi_transcripts,
        signal_integrity,
        signal_strength,
        x,
        y,
        window_size,
        signal_threshold=signal_threshold,
        figsize=figsize,
    )


# top-bottom consistency


def celltype_consistency(obs1, obs2, levels):
    import pandas as pd
    from scipy.stats.contingency import crosstab

    occurences = crosstab(obs1, obs2, levels=[levels, levels])

    combinations = pd.DataFrame(
        occurences.count, index=occurences.elements[0], columns=occurences.elements[1]
    )

    # make symmetric
    combinations = combinations + combinations.T
    combinations /= np.eye(len(combinations)) + 1

    counts = combinations.sum().astype(int)

    # normalize
    combinations = combinations.div(combinations.sum(axis="rows"), axis="rows")
    return combinations, counts


def plot_celltype_consistency(
    matrix, counts, figsize=None, cmap="turbo", f=None, offset=3.5, fig_ax=None
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    heatmap_kwargs = dict(
        cmap=cmap,
        fmt="",
        vmin=0,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        square=True,
    )

    if f is not None:
        anno = pd.DataFrame("", index=matrix.index, columns=matrix.index, dtype=str)
        for i in range(len(matrix.index)):
            anno.iloc[i, i] = f"{matrix.iloc[i, i]:{f}}"

        heatmap_kwargs["annot"] = anno

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_ax

    sns.heatmap(
        matrix,
        ax=ax,
        **heatmap_kwargs,
        annot_kws={"size": 6},
        cbar_kws={"label": "Proportion of observations"},
    )

    ax.set(xlabel="Cell type 2", ylabel="Cell type 1 (normalized)")

    # count annotation
    for y, count in enumerate(counts):
        ax.text(matrix.shape[1] + offset, y + 0.5, str(count), va="center", ha="right")
    ax.text(matrix.shape[1] + offset, -0.3, "# cells", va="bottom", ha="right")

    # border around colorbar/heatmap
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)
            spine.set_visible(True)

    return fig
