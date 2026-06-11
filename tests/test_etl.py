# tests/test_etl.py


def test_csv_loads_correctly():
    import pandas as pd

    df = pd.read_csv("data/report.csv")
    assert len(df) > 0  # file is not empty
    assert "label" in df.columns  # expected column exists


def test_no_missing_values():
    import pandas as pd

    df = pd.read_csv("data/report.csv")
    assert df.isnull().sum().sum() == 0  # no NaN anywhere
