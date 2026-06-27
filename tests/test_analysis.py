import numpy as np
import pandas as pd
from scripts.analysis import correlation_table


def test_correlation_table_perfect():
    df = pd.DataFrame({
        "trait_rate": [0.1, 0.2, 0.3, 0.4],
        "eas_last": [0.1, 0.2, 0.3, 0.4],
        "divergence_rate_B": [0.4, 0.3, 0.2, 0.1],
    })
    out = correlation_table(df, ["eas_last", "divergence_rate_B"])
    row_eas = out.set_index("metric").loc["eas_last"]
    assert np.isclose(row_eas["pearson_r"], 1.0)
    row_div = out.set_index("metric").loc["divergence_rate_B"]
    assert np.isclose(row_div["pearson_r"], -1.0)
