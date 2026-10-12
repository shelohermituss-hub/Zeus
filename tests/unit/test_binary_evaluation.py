"""Tests des outils d'évaluation des études options binaires."""
import pytest

from zeus.backtest.binary_engine import wilson_lower_bound
from zeus.backtest.binary_evaluation import Tally, pip_size, pool


@pytest.mark.parametrize("symbol,size", [("eurusd", 1e-4), ("USDJPY", 0.01), ("xauusd", 0.1)])
def test_pip_size(symbol, size):
    assert pip_size(symbol) == size


def test_wilson_bound_below_rate_and_tightens_with_n():
    assert wilson_lower_bound(0, 0) == 0.0
    small, large = wilson_lower_bound(60, 100), wilson_lower_bound(6000, 10000)
    assert small < large < 0.6


def test_pool_aggregates_and_counts_datasets_above_breakeven():
    stats = pool({"a": Tally(60, 40), "b": Tally(40, 60), "c": Tally(0, 0)}, breakeven=0.52)
    assert stats["trades"] == 200 and stats["win_rate"] == 0.5
    assert stats["share_above_breakeven"] == 0.5
    assert "c" not in stats["per_dataset_win_rate"]
