"""Datasets shipped with impute."""

from importlib.resources import files


def load_khanmiss():
    """Khan microarray data with random missing values (R's ``khanmiss``).

    A DataFrame of strings with 2309 rows and 65 columns, laid out exactly
    like the R data.frame: row 0 has the class labels, the remaining rows are
    gene expression. Column ``X`` is a dummy gene number, column ``X1`` the
    gene name, and the remaining columns are gene expression. Missing values
    are NaN.

    The R example ``as.matrix(khanmiss[-1, -(1:2)])`` corresponds to
    ``load_khanmiss().iloc[1:, 2:].to_numpy(dtype=float)``.

    Requires pandas (``pip install biocimpute[pandas]``).

    Khan, J. et al. (2001) Classification and diagnostic prediction of
    cancers using gene expression profiling and artificial neural networks.
    Nature Medicine 7, 673-679.
    """
    import pandas as pd

    with files(__name__).joinpath("khanmiss.csv.gz").open("rb") as f:
        return pd.read_csv(
            f,
            compression="gzip",
            dtype=str,
            keep_default_na=False,
            na_values=["NA"],
        )
