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

    import ovrlpy

    coordinate_df = ovrlpy.io.read_Xenium(args.input)

    _ = ovrlpy.run(
        df=coordinate_df,
        cell_diameter=10,
        n_expected_celltypes=args.n_pcs,
        n_workers=args.n_threads,
    )


if __name__ == "__main__":
    main()
