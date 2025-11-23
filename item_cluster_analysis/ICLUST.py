import numpy as np
import pandas as pd
from typing import Dict, List, Any

# --- Helper Functions ---
def cronbach_alpha_from_corr(R_sub: np.ndarray) -> float:
    k = R_sub.shape[0]
    if k <= 1:
        return 1.0

    vals = np.asarray(R_sub)
    s = vals.sum() - np.trace(vals)
    rbar = s / (k * (k - 1))
    denominator = 1 + (k - 1) * rbar
    alpha = (k * rbar) / denominator

    return max(0.0, alpha)

def revelle_beta_criterion(r12):
    if r12 <= -1:
        return np.nan
    return (2 * r12) / (1 + r12)

def cluster_cor(keys, R):
    keys = np.asarray(keys, dtype=float)
    item_covar = R.values @ keys
    covar = keys.T @ item_covar
    var = np.diag(covar)
    sd = np.sqrt(var + 1e-12)
    cor = covar / (sd[:, None] * sd[None, :])
    size = np.diag(keys.T @ keys).astype(int)
    alphas = []
    for j in range(keys.shape[1]):
        items_j = list(R.index[np.abs(keys[:, j]) > 0])
        alphas.append(cronbach_alpha_from_corr(R.loc[items_j, items_j]))
    alphas = np.array(alphas, dtype=float)
    cor_df = pd.DataFrame(cor, index=[f"K{j+1}" for j in range(keys.shape[1])],
                          columns=[f"K{j+1}" for j in range(keys.shape[1])])
    return {"cor": cor_df, "alpha": alphas, "size": size}

def rbar_from_alpha(alpha, k):
    denominator = k - alpha * (k - 1)
    return alpha / denominator if denominator != 0 else np.nan

# --- ICLUST Clustering ---
def iclust_cluster(R, n_clusters=2, alpha_rule=3, beta_rule=1,
                   beta_size=4, alpha_size=3, correct=True, reverse=True,
                   beta_min=0.5, verbose=True) -> Dict[str, Any]:
    if not isinstance(R, pd.DataFrame):
        R = pd.DataFrame(R)

    p = R.shape[0]
    keys = np.eye(p, dtype=float)
    cluster_names = [f"V{i+1}" for i in range(p)]
    prev_stats = {}
    results_rows = []
    keep_clustering = True
    count = 1

    while keep_clustering:
        m = keys.shape[1]
        stats = cluster_cor(keys, R)
        sim_mat = stats["cor"].copy()
        np.fill_diagonal(sim_mat.values, 0.0)

        if correct:
            row_min = sim_mat.min(axis=1).values
            row_maxv = sim_mat.max(axis=1).values
            row_max_abs = np.maximum(np.abs(row_min), np.abs(row_maxv))
        else:
            row_max_abs = np.ones(m)

        # use strongest absolute similarity if the cluster only contains one item
        item_rel = stats["alpha"].copy()
        for i in range(m):
            if stats["size"][i] < 2:
                item_rel[i] = row_max_abs[i]

        # Create corrected similarity matrix
        sq_max = np.diag(1 / np.sqrt(item_rel + 1e-12))
        sim = sq_max @ sim_mat.values @ sq_max if correct else sim_mat.values.copy()
        np.fill_diagonal(sim, np.nan)

        # serach for best merge
        test_alpha = False
        test_beta = False

        while not (test_alpha and test_beta):
            if np.all(np.isnan(sim)):
                keep_clustering = False
                break

            max_cell = np.nanargmax(sim)
            max_row = max_cell % m
            max_col = max_cell // m

            sign_max = 1.0
            if reverse:
                min_cell = np.nanargmin(sim)
                min_row = min_cell % m
                min_col = min_cell // m
                if sim[max_row, max_col] < abs(sim[min_row, min_col]):
                    sign_max = -1.0
                    max_row, max_col = min_row, min_col
                if sim[max_row, max_col] < 0:
                    sign_max = -1.0

            r12_raw = sim_mat.values[max_row, max_col]
            r12 = sign_max * r12_raw

            size1 = stats["size"][max_row]
            size2 = stats["size"][max_col]
            name1 = cluster_names[max_row]
            name2 = cluster_names[max_col]

            if size1 < 2:
                alpha1 = item_rel[max_row]
                beta1 = item_rel[max_row]
                rbar1 = item_rel[max_row]
            else:
                ps = prev_stats[name1]
                alpha1, beta1, rbar1 = ps["alpha"], ps["beta"], ps["rbar"]

            if size2 < 2:
                alpha2 = item_rel[max_col]
                beta2 = item_rel[max_col]
                rbar2 = item_rel[max_col]
            else:
                ps = prev_stats[name2]
                alpha2, beta2, rbar2 = ps["alpha"], ps["beta"], ps["rbar"]

            V1 = size1 + size1 * (size1 - 1) * rbar1
            V2 = size2 + size2 * (size2 - 1) * rbar2
            Cov12 = r12 * np.sqrt(V1 * V2)
            V12 = V1 + V2 + 2 * Cov12
            size12 = size1 + size2

            alpha_new = ((V12 - size12) * (size12 / (size12 - 1))) / (V12 + 1e-12)
            rbar_new = rbar_from_alpha(alpha_new, size12)
            beta_new = revelle_beta_criterion(r12)

            test_alpha = True
            if alpha_size < min(size1, size2):
                if alpha_rule == 1:
                    if alpha_new < min(alpha1, alpha2): 
                        test_alpha = False
                elif alpha_rule == 2:
                    if alpha_new < (alpha1 + alpha2) / 2: 
                        test_alpha = False
                else:
                    if alpha_new < max(alpha1, alpha2): 
                        test_alpha = False

            test_beta = True
            if beta_size < min(size1, size2):
                if beta_rule == 1:
                    if beta_new < min(beta1, beta2): 
                        test_beta = False
                elif beta_rule == 2:
                    if beta_new < (beta1 + beta2) / 2: 
                        test_beta = False
                else:
                    if beta_new < max(beta1, beta2): 
                        test_beta = False

            if test_alpha and test_beta:
                break
            else:
                if np.isnan(beta_new) or beta_new < beta_min:
                    keep_clustering = False
                    break
                sim[max_row, max_col] = np.nan
                sim[max_col, max_row] = np.nan

        if not keep_clustering:
            break

        if max_col < max_row:
            max_row, max_col = max_col, max_row

        keys[:, max_row] = keys[:, max_row] + sign_max * keys[:, max_col]
        merged_items = list(R.index[np.abs(keys[:, max_row]) > 0])

        keys = np.delete(keys, max_col, axis=1)
        cluster_names.pop(max_col)
        new_name = f"C{count}"
        cluster_names[max_row] = new_name

        prev_stats[new_name] = {
            "alpha": float(alpha_new),
            "beta": float(beta_new),
            "rbar": float(rbar_new),
            "size": int(size12)
        }

        results_rows.append({
            "step": count,
            "merged_left": name1,
            "merged_right": name2,
            "r12_raw": float(r12_raw),
            "alpha_new": float(alpha_new),
            "beta_new": float(beta_new),
            "size_new": int(size12),
            "items": merged_items
        })

        if verbose:
            print(f"step {count}: merge {name1} + {name2} -> {new_name} "
                  f"| r12={r12_raw:.3f}, alpha={alpha_new:.3f}, beta={beta_new:.3f}, size={size12} "
                  f"| items={merged_items}")

        count += 1
        if (p - count) < n_clusters or keys.shape[1] <= 1:
            keep_clustering = False

    return {"results": pd.DataFrame(results_rows),
            "clusters": keys,
            "cluster_names": cluster_names}


    

