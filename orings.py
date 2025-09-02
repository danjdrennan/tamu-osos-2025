"""challenger_orings: Analysis for the NASA Challenger O-rings dataset [1,2].

This is a minimal program demonstrating a machine learning pipeline for a fixed problem
and configuration. The data are pre-loaded into the osos library so we can avoid IO
anywhere in the program.

References:
    [1] Background: https://en.wikipedia.org/wiki/Space_Shuttle_Challenger_disaster
    [2] Data accessed from https://www.openintro.org/data/index.php?data=orings
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model

import osos

# Constants of a program should be defined immediately after imports. They may be single
# constants like this or wrapped in a dataclass.
SAVE_FIG = True
FIG_FILENAME = "challenger_orings.png"

if SAVE_FIG:
    FIG_SIZE = (6.5, 3)
    FONT_SIZES = {"title": 12, "axes": 10, "legend": 6}
    FIG_DPI = 300
else:
    FIG_SIZE = (13, 6)
    FONT_SIZES = {"title": 16, "axes": 14, "legend": 10}


def main():
    """Always fence the executable region of a program off in main.

    Putting the main execution in a main function like this means it can be imported to
    another program without eagerly executing upon import. This version does not allow
    any configurability of the constants defined above (SAVE_FIG, etc). We demonstrate
    how to make this more portable with greater configurability in `app.py`.
    """
    ds, _ = osos.load_dataset(osos.Datasets.CHALLENGER)

    model = linear_model.LogisticRegression(
        penalty=None,  # type: ignore
        fit_intercept=True,
        max_iter=1000,
    )
    model.fit(ds.x, ds.y)

    assert isinstance(model.intercept_, np.ndarray)
    b0 = model.intercept_[0]
    b1 = model.coef_[0, 0]
    odds_ratio = np.exp(b1)
    print(
        f"""Model fit:
          {"intercept":<16}{b0:>12.4f}
          {"temperature":<16}{b1:>12.4f}
          {"odds ratio":<16}{odds_ratio:>12.4f}
          """
    )

    x = np.arange(40, 90).astype(np.float64).reshape(-1, 1)
    predictions = model.predict_proba(x)

    # The predictions return with shape (num_obs, num_classes). We want the second dim of
    # num_classes as it represents the predictions for an o-ring failing
    predictions = predictions[:, 1]

    launch_temp = 36.0
    p_fail = model.predict_proba(np.array(launch_temp).reshape(1, 1))[0, 1]

    print(f"Launch temperature (T): {launch_temp} \u00b0F")
    print(f"P(Failure | T={launch_temp} \u00b0F) = {p_fail:.5f}")

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, layout="constrained")

    ax.plot(ds.x.squeeze(), ds.y.squeeze(), "ko", label="Launches")
    ax.plot(x.squeeze(), predictions, "C0", label="p(failure | temp)")
    ax.axvline(
        launch_temp, color="k", linestyle="--", label="Challenger launch temperature"
    )

    ax.set_title("Challenger O-Ring failures", fontsize=FONT_SIZES["title"])
    ax.set_xlabel("Temperature (\u00b0F)", fontsize=FONT_SIZES["axes"])
    ax.set_ylabel("O-Ring failed", fontsize=FONT_SIZES["axes"])
    ax.legend(fontsize=FONT_SIZES["legend"])

    # This lets us fence off code we don't always want to run
    if SAVE_FIG:
        fig.savefig(FIG_FILENAME, dpi=FIG_DPI, pad_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
