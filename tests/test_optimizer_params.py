import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from skopt.space import Integer, Categorical

from optimize_hermes import optimize_parameters_bayesian


class DummyResult:
    def __init__(self, x, fun):
        self.x = x
        self.fun = fun


def _build_dummy_data(days: int = 400) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=days, freq="D")
    base = 100 + np.linspace(0, 20, days)
    noise = np.sin(np.linspace(0, 12, days)) * 2
    close = base + noise
    high = close * 1.01
    low = close * 0.99
    open_ = close * 0.995

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        },
        index=index,
    )


class OptimizerParameterTests(unittest.TestCase):
    def test_best_params_contains_baseline_and_macro_keys(self):
        data = _build_dummy_data()
        start_date = data.index[0]
        end_date = data.index[-1]

        param_space = [
            Integer(15, 16, name="short_period"),
            Integer(40, 42, name="long_period"),
            Integer(90, 91, name="alma_offset_int"),
            Categorical([3.0], name="alma_sigma"),
            Categorical([3], name="buy_momentum_bars"),
            Categorical([0], name="sell_momentum_bars"),
            Categorical([3], name="baseline_momentum_bars"),
            Categorical([200], name="macro_ema_period"),
        ]

        sample_params = {
            "short_period": 15,
            "long_period": 40,
            "alma_offset_int": 90,
            "alma_sigma": 3.0,
            "buy_momentum_bars": 3,
            "sell_momentum_bars": 0,
            "baseline_momentum_bars": 3,
            "macro_ema_period": 200,
        }

        def fake_gp_minimize(func, dimensions, n_calls, n_random_starts, random_state, verbose, n_jobs):
            values = [sample_params[dim.name] for dim in dimensions]
            objective_value = func(values)
            return DummyResult(values, objective_value)

        with patch("optimize_hermes.gp_minimize", side_effect=fake_gp_minimize):
            best = optimize_parameters_bayesian(
                data,
                start_date,
                end_date,
                param_space,
                n_calls=2,
                n_random_starts=1,
                n_jobs=1,
            )

        self.assertIn("baseline_momentum_bars", best)
        self.assertIn("macro_ema_period", best)
        self.assertEqual(best["baseline_momentum_bars"], sample_params["baseline_momentum_bars"])
        self.assertEqual(best["macro_ema_period"], sample_params["macro_ema_period"])
        self.assertGreater(best["alma_offset"], 0.0)
        self.assertLessEqual(best["alma_offset"], 1.0)


if __name__ == "__main__":
    unittest.main()
