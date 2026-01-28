import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr, pearsonr
from typing import List, Union, Any, Optional

# --- Helper Functions for Correlation Measures ---
def cramers_v(x: Union[pd.Series, List[Any]], y: Union[pd.Series, List[Any]]) -> float:
    """
    Computes Cramer's V for two categorical variables.

    Args:
        x: A list, array, or pandas Series of categorical data.
        y: A list, array, or pandas Series of categorical data.

    Returns:
        The Cramer's V statistic (float) or np.nan if the data is insufficient.
    """
    x = pd.Series(x)
    y = pd.Series(y)
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    table = pd.crosstab(x, y)
    n = table.values.sum()

    if n == 0:
        return np.nan

    chi2, _, _, _ = chi2_contingency(table)
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))

    if denom <= 0:
        return np.nan
        
    return np.sqrt(phi2corr / denom)


def correlation_ratio(categories: Union[pd.Series, List[Any]], values: Union[pd.Series, List[Any]]) -> float:
    """
    Computes the Correlation Ratio (Eta) for an association between a
    categorical variable and a numerical/ordinal variable.

    Args:
        categories: A list, array, or pandas Series of categorical data.
        values: A list, array, or pandas Series of numerical/ordinal data.

    Returns:
        The Correlation Ratio (float) in the range [0, 1] or np.nan.
    """
    categories = pd.Series(categories)
    values = pd.Series(values)
    mask = categories.notna() & values.notna()
    categories = categories[mask]
    values = values[mask].astype(float)

    if len(values) < 2:
        return np.nan

    grand_mean = values.mean()
    groups = [values[categories == c] for c in categories.unique()]

    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups if len(g) > 0)
    ss_total = ((values - grand_mean) ** 2).sum()

    if ss_total == 0:
        return 0.0

    return np.sqrt(ss_between / ss_total)


# --- Correlation Functions for Specific Pairings ---

def compute_ordinal_ordinal_corr(x: Union[pd.Series, List[Any]], y: Union[pd.Series, List[Any]]) -> float:
    """
    Computes Spearman's Rank Correlation for two ordinal variables.
    """
    return spearmanr(x, y, nan_policy='omit').correlation

def compute_ordinal_numerical_corr(x: Union[pd.Series, List[Any]], y: Union[pd.Series, List[Any]]) -> float:
    """
    Computes Spearman's Rank Correlation for a mixed ordinal-numerical pair.
    """
    return spearmanr(x, y, nan_policy='omit').correlation

def compute_ordinal_categorical_corr(x: Union[pd.Series, List[Any]], y: Union[pd.Series, List[Any]]) -> float:
    """
    Computes Correlation Ratio (Eta) between an ordinal variable (x) and a
    categorical variable (y).
    """
    x_series = pd.Series(x)
    y_series = pd.Series(y)

    if pd.api.types.is_numeric_dtype(x_series.dtype):
        ord_vals = x_series.astype(float)
    else:
        ord_vals = x_series.astype("category").cat.codes.astype(float)

    return correlation_ratio(y_series, ord_vals)

def compute_numerical_categorical_corr(x: Union[pd.Series, List[Any]], y: Union[pd.Series, List[Any]]) -> float:
    """
    Computes Correlation Ratio (Eta) between a numerical variable (x) and a
    categorical variable (y).
    """
    return correlation_ratio(y, pd.Series(x).astype(float))

def compute_numerical_numerical_corr(x: Union[pd.Series, List[Any]], y: Union[pd.Series, List[Any]]) -> float:
    """
    Computes Pearson's Linear Correlation for two numerical variables.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return np.nan
        
    r, _ = pearsonr(x, y)
    return r

def compute_categorical_categorical_corr(x: Union[pd.Series, List[Any]], y: Union[pd.Series, List[Any]]) -> float:
    """
    Computes Cramer's V for two categorical/nominal variables.
    """
    return cramers_v(x, y)

# --- Main Matrix Function ---

def compute_mixed_corr_matrix(
    df: pd.DataFrame,
    categorical_vars: Optional[List[str]] = None,
    numerical_vars: Optional[List[str]] = None,
    ordinal_vars: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Computes a correlation matrix for a DataFrame containing a mix of
    numerical, ordinal, and categorical variables using appropriate measures.

    The correlation measures used are:
    - Numerical-Numerical: Pearson's r
    - Ordinal-Ordinal: Spearman's rho
    - Numerical/Ordinal-Categorical: Correlation Ratio (Eta)
    - Categorical-Categorical: Cramer's V
    - Numerical-Ordinal: Spearman's rho

    Args:
        df: The input pandas DataFrame.
        categorical_vars: A list of column names for categorical variables.
        numerical_vars: A list of column names for continuous numerical variables.
        ordinal_vars: A list of column names for ordinal variables.

    Returns:
        A pandas DataFrame representing the mixed-type correlation matrix.
    """
    categorical_vars = categorical_vars or []
    numerical_vars = numerical_vars or []
    ordinal_vars = ordinal_vars or []

    if not (categorical_vars or numerical_vars or ordinal_vars):
        raise ValueError("At least one variable list must be non-empty.")

    overlap = (
        set(categorical_vars) & set(numerical_vars)
        | set(categorical_vars) & set(ordinal_vars)
        | set(numerical_vars) & set(ordinal_vars)
    )
    if overlap:
        raise ValueError(f"Variables assigned to multiple types: {overlap}")

    vars_all = categorical_vars + numerical_vars + ordinal_vars
    k = len(vars_all)
    
    corr = pd.DataFrame(np.nan, index=vars_all, columns=vars_all)

    # Compute correlations
    for i in range(k):
        v1 = vars_all[i]
        for j in range(i, k):
            v2 = vars_all[j]

            # 1. Diagonal
            if v1 == v2:
                r = 1.0
            else:
                # 2. Variable Pairings
                v1_cat = 'num' if v1 in numerical_vars else 'ord' if v1 in ordinal_vars else 'cat'
                v2_cat = 'num' if v2 in numerical_vars else 'ord' if v2 in ordinal_vars else 'cat'
                
                pair = tuple(sorted((v1_cat, v2_cat)))
                
                data_v1 = df[v1]
                data_v2 = df[v2]
                
                
                if pair == ('num', 'num'):
                    r = compute_numerical_numerical_corr(data_v1, data_v2)
                elif pair == ('ord', 'ord'):
                    r = compute_ordinal_ordinal_corr(data_v1, data_v2)
                elif pair == ('cat', 'cat'):
                    r = compute_categorical_categorical_corr(data_v1, data_v2)
                elif pair == ('num', 'ord'):
                    r = compute_ordinal_numerical_corr(data_v1, data_v2)
                elif pair == ('cat', 'ord'):
                    if v1_cat == 'ord':
                        r = compute_ordinal_categorical_corr(data_v1, data_v2)
                    else:
                        r = compute_ordinal_categorical_corr(data_v2, data_v1)
                elif pair == ('cat', 'num'):
                    if v1_cat == 'num':
                        r = compute_numerical_categorical_corr(data_v1, data_v2)
                    else:
                        r = compute_numerical_categorical_corr(data_v2, data_v1)
                else:
                    r = np.nan

            corr.loc[v1, v2] = r
            corr.loc[v2, v1] = r

    return corr
