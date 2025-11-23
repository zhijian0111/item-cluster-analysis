# Item Cluster Analysis (ICLUST) in Python

A Python package providing:

- **Mixed-type correlation matrix computation**
- **ICLUST hierarchical item clustering** (Python translation of the ICLUST algorithm from the R `psych` package)

> The ICLUST implementation is adapted from the R version documented at:  
> https://personality-project.org/r/r.ICLUST.html

---

## Features

### Mixed-Type Correlation Matrix

Implemented in `correlation_computation.py`, using the correct measure for each variable pairing:

| Variable Pair | Correlation Method |
|---------------|--------------------|
| Numerical ↔ Numerical | Pearson's *r* |
| Ordinal ↔ Ordinal | Spearman's *ρ* |
| Numerical ↔ Ordinal | Spearman's *ρ* |
| Categorical ↔ Numerical / Ordinal | Correlation Ratio (*η*) |
| Categorical ↔ Categorical | Cramér's V |

---

### ICLUST Hierarchical Clustering

Implemented in `ICLUST.py`.

Outputs:
- **`results`** — merge history (step-by-step)
- **`clusters`** — final cluster membership matrix
- **`cluster_names`** — labels of the final clusters

> **Note:**  
> The ICLUST algorithm may stop *before* reaching the requested `n_clusters` if  
> the alpha/beta merge criteria reject all remaining merges.  
> In this case, the final number of clusters will be larger than `n_clusters`.  


---

## Installation

```bash
pip install git+https://github.com/zhijian0111/item-cluster-analysis.git
