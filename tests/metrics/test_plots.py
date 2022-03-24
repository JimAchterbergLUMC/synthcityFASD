# stdlib
from typing import Callable

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# synthcity absolute
from synthcity.metrics.plots import (
    plot_associations_comparison,
    plot_marginal_comparison,
)
from synthcity.plugins import Plugin, Plugins


def _eval_plugin(cbk: Callable, X: pd.DataFrame, X_syn: pd.DataFrame) -> None:
    cbk(plt, X, X_syn)

    sz = len(X_syn)
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)

    cbk(
        plt,
        X,
        X_rnd,
    )


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_plot_marginal_comparison(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    _eval_plugin(plot_marginal_comparison, X, X_gen)


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_plot_associations_comparison(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    test_plugin.fit(X)
    X_gen = test_plugin.generate(1000)

    _eval_plugin(plot_associations_comparison, X, X_gen)
