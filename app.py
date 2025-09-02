"""app: Analysis for the NASA Challenger O-rings dataset [1,2].

The program uses `sklearn.linear_model.LogisticRegression` with no regularization to
model the Challenger O-rings dataset. The results are printed on the command line and
plotted using a GUI.

Results are printed in two places: A set of diagnostics are printed on `stderr`, the
standard error file stream. The results of the analysis are printed on `stdout` as a
csv, which can be piped into other programs or saved.

References:
    [1] Background: https://en.wikipedia.org/wiki/Space_Shuttle_Challenger_disaster
    [2] Data accessed from https://www.openintro.org/data/index.php?data=orings
"""

# Now we want to provide optionality for calling the program with parameters. To do
# this, we are going to use some of Python's standard library. We want support for
# argument parsing to read command line args, the sys module to control what file stream
# results are printed on, and a dataclass to create a better container of our config.
import argparse
import sys
import textwrap
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model

import osos


# Still defining what will be constants, but we've deferred their setup to `get_config`.
# Now this config is exposing all of the knobs one could turn in a flat data structure.
@dataclass(frozen=True)
class Config:
    """Configurable runtime arguments assigned via CLI args."""

    fig_axes_fontsize: int | float
    fig_dpi: int | None
    fig_filename: str | None
    fig_legend_fontsize: int | float
    fig_title_fontsize: int | float
    figsize: tuple[int | float, ...]
    plot_fig: bool
    save_fig: bool


def main(args: list[str] | None = None):
    f"""{__doc__}"""

    config = get_config(args)

    # Printing here is done on "standard error", a file stream that is useful for
    # communicating the status of a program without clobbering the results that should
    # be printed on "standard out". These streams and the "standard in" stream are three
    # standard streams that any program can rely on to exist in a Unix-like system. More
    # info on these streams may be found at
    #
    #   [Wikipedia Standard Streams](https://en.wikipedia.org/wiki/Standard_streams)
    #
    print(f"{config}\n", file=sys.stderr)

    ds, _ = osos.load_dataset(osos.Datasets.CHALLENGER)

    model = linear_model.LogisticRegression(
        penalty=None,  # type: ignore
        fit_intercept=True,
        max_iter=1000,
    )
    model.fit(ds.x, ds.y)

    # Start: collecting results into objects
    assert isinstance(model.intercept_, np.ndarray)
    intercept = model.intercept_[0].item()

    coef = model.coef_[0, 0].item()
    odds_ratio = np.exp(coef).item()
    train_score = model.score(ds.x, ds.y)

    results = dict(
        intercept=intercept, coef=coef, odds_ratio=odds_ratio, train_score=train_score
    )

    out_str = textwrap.dedent(
        f"""\
        Model fit:
          {"intercept":<16}{intercept:>12.4f}
          {"temperature":<16}{coef:>12.4f}
          {"odds ratio":<16}{odds_ratio:>12.4f}\
        """
    )
    print(f"{out_str}\n", file=sys.stderr)

    x = np.arange(40, 90).astype(np.float64).reshape(-1, 1)
    predictions = model.predict_proba(x)

    # The predictions return with shape (num_obs, num_classes). We want the second dim of
    # num_classes as it represents the predictions for an o-ring failing
    predictions = predictions[:, 1]

    launch_temp = 36.0
    p_fail = model.predict_proba(np.array(launch_temp).reshape(1, 1))[0, 1].item()

    results["launch_temp"] = launch_temp
    results["launch_temp_failure_prob"] = p_fail

    out_str = textwrap.dedent(
        f"""\
        Launch temperature (T): {launch_temp}\u00b0F
        P(Failure | T={launch_temp}\u00b0F) = {p_fail:.5f}\
        """
    )
    print(f"{out_str}\n", file=sys.stderr)

    # Table: Printing table results to csv
    n = len(results.keys())
    for i, k in enumerate(results.keys()):
        p = f"{k}," if i + 1 < n else f"{k}"
        print(p, end="", file=sys.stdout)
    print("")
    for i, v in enumerate(results.values()):
        p = f"{v:.5f}," if i + 1 < n else f"{v:.5f}"
        print(p, end="", file=sys.stdout)
    print("\n")

    # Plotting
    if config.plot_fig or config.save_fig:
        # This gets skipped if both flags are set to False. Otherwise we need to produce
        # a figure following the user inputs.
        fig, ax = plt.subplots(1, 1, figsize=config.figsize, layout="constrained")

        ax.plot(ds.x.squeeze(), ds.y.squeeze(), "ko", label="Launches")
        ax.plot(x.squeeze(), predictions, "C0", label="p(failure | temp)")
        ax.axvline(
            launch_temp,
            color="k",
            linestyle="--",
            label="Challenger launch temperature",
        )

        ax.set_title("Challenger O-Ring failures", fontsize=config.fig_title_fontsize)
        ax.set_xlabel("Temperature (\u00b0F)", fontsize=config.fig_axes_fontsize)
        ax.set_ylabel("O-Ring failed", fontsize=config.fig_axes_fontsize)
        ax.legend(fontsize=config.fig_legend_fontsize)

        # Now we just check the flags and do whatever was requested. Note here that we
        # need to be certain a value was provided in the fig filename.
        if config.save_fig:
            fname = config.fig_filename or "challenger_orings.png"
            fig.savefig(fname, dpi=config.fig_dpi, pad_inches="tight")

        if config.plot_fig:
            plt.show()

    exit(0)


def get_config(args: list[str] | None = None) -> Config:
    """Parses CLI or other args list to produce a config.

    Minimal validation to choose good defaults for plots is provided. This is extracted
    from the `main` function because it is cumbersome and contributes nothing
    substantial to the program (this functions adds 100 lines of code to the program in
    order to expose a few plotting arguments).
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--plot-fig",
        action="store_true",
        default=False,
        help="Plot figure [requires opt-in]",
    )

    parser.add_argument(
        "--save-fig",
        action="store_true",
        default=False,
        help="Save figure [requires opt-in]",
    )

    parser.add_argument(
        "--fig-filename",
        type=str,
        default="challenger_orings.png",
        help="filename to save to",
    )

    parser.add_argument(
        "--fig-dpi",
        type=int,
        default=300,
        help="The dots per inch of the saved png [default is min recommendation]",
    )

    parser.add_argument(
        "--figsize",
        type=tuple,
        default=(13, 6),
        help=(
            "Figure size defaults to interactive plot sizes. This size is not ideal "
            "for papers or posters. There the recommendation is (6.5, 3), where the "
            "units are in inches."
        ),
    )

    parser.add_argument(
        "--fig-title-fontsize",
        type=float,
        default=16.0,
        help="Default for the (13,6) figsize. Recommend 12 for a (6.5, 3) figure.",
    )

    parser.add_argument(
        "--fig-axes-fontsize",
        type=float,
        default=14.0,
        help="Default for (13,6) figszie. Recommend 10 for a (6.5, 3) figure.",
    )

    parser.add_argument(
        "--fig-legend-fontsize",
        type=float,
        default=10.0,
        help="Default for (13,6) figsize. Recommend 6 for a (6.5, 3) figure.",
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.save_fig:
        fig_axes_fontsize = parsed_args.fig_axes_fontsize or 10
        fig_dpi = parsed_args.fig_dpi or 300
        fig_filename = parsed_args.fig_filename or "challenger_orings.png"
        fig_legend_fontsize = parsed_args.fig_legend_fontsize or 6
        fig_title_fontsize = parsed_args.fig_title_fontsize or 12
        figsize = parsed_args.figsize or (6.5, 3)
    else:
        fig_axes_fontsize = parsed_args.fig_axes_fontsize or 14
        fig_dpi = None
        fig_filename = None
        fig_legend_fontsize = parsed_args.fig_legend_fontsize or 10
        fig_title_fontsize = parsed_args.fig_title_fontsize or 16
        figsize = parsed_args.figsize or (13, 6)

    return Config(
        fig_axes_fontsize=fig_axes_fontsize,
        fig_dpi=fig_dpi,
        fig_filename=fig_filename,
        fig_legend_fontsize=fig_legend_fontsize,
        fig_title_fontsize=fig_title_fontsize,
        figsize=figsize,
        plot_fig=parsed_args.plot_fig,
        save_fig=parsed_args.save_fig,
    )


if __name__ == "__main__":
    SystemExit(main())
