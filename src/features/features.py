import pandas as pd


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()

    # Total square footage (above and below ground)
    df["total_sf"] = df["1st Flr SF"] + df["2nd Flr SF"] + df["Total Bsmt SF"]

    # Total bathrooms with basement adjustment
    df["bathrooms"] = (
        df["Full Bath"] + 0.5 * df["Half Bath"] +
        df["Bsmt Full Bath"] + 0.5 * df["Bsmt Half Bath"]
    )

    # House age
    df["house_age"] = df["Yr Sold"] - df["Year Built"]

    # Years since remodel
    df["since_remodel"] = df["Yr Sold"] - df["Year Remod/Add"]

    # Treat MS SubClass as a categorical string
    df["MS SubClass"] = df["MS SubClass"].astype(str)

    return df
