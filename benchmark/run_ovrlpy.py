#!/usr/bin/env python


def main():
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description="")

    parser.add_argument("input", help="Path to input file.", type=Path)
    parser.add_argument(
        "--n_threads",
        help="Number of threads for parallelization.",
        type=int,
        default=8,
    )
    parser.add_argument("--n_pcs", help="Number of PCs.", type=int, default=30)

    args = parser.parse_args()

    import os

    os.environ["POLARS_MAX_THREADS"] = f"{args.n_threads}"

    import ovrlpy

    analysis = ovrlpy.Ovrlp(
        ovrlpy.io.read_Xenium(args.input),
        n_components=args.n_pcs,
        n_workers=args.n_threads,
    )
    analysis.analyse()

    print("Done")


if __name__ == "__main__":
    main()
