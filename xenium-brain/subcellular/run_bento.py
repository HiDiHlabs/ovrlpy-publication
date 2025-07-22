#!/usr/bin/env python


def main():
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description="")

    parser.add_argument("input", help="Path to input file.", type=Path)
    parser.add_argument("output", help="Path to output file (pickle).", type=Path)
    parser.add_argument(
        "--n_threads",
        help="Number of threads for parallelization.",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    import bento as bt
    import spatialdata_io

    sdata = spatialdata_io.xenium(
        args.input,
        cells_labels=False,
        nucleus_labels=False,
        cells_as_circles=False,
        morphology_mip=False,
        morphology_focus=False,
        aligned_images=False,
        n_jobs=args.n_threads,
    )

    # remove index name otherwise bento fails
    for df in sdata.shapes.values():
        df.rename_axis(index=None, inplace=True)

    # drop control genes
    df = sdata.points["transcripts"].loc[
        lambda df: ~df["feature_name"].str.startswith(("BLANK", "NegControl"))
    ]
    df["feature_name"] = df["feature_name"].cat.remove_unused_categories()
    sdata.points["transcripts"] = df

    # bento
    sdata = bt.io.prep(sdata)

    bt.tl.lp(sdata, num_workers=args.n_threads)

    # writing sdata to zarr fails
    import pickle

    with open(args.output, "wb") as file:
        pickle.dump(sdata, file)

    print("Done")


if __name__ == "__main__":
    main()
