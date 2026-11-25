"""Research methodology advisors: statistical tests and figure type recommendations."""

import warnings

import numpy as np
from scipy import stats

# Below this sample size a normality test has too little power to be useful;
# such samples are assumed normal (and reported as "n too small").
_MIN_NORMALITY_N = 8
# Shapiro-Wilk is used below this size, D'Agostino-Pearson above it.
_SHAPIRO_MAX_N = 5000
_ALPHA = 0.05


def _format_p(p: float) -> str:
    return "p<0.0001" if p < 1e-4 else f"p={p:.4f}"


def _normality(sample: np.ndarray, alpha: float = _ALPHA) -> dict:
    """Test one sample for normality at significance level ``alpha``.

    Returns dict with test, scipy, p_value (None if not tested), is_normal and
    constant (zero variance, where a normality test is undefined).
    """
    n = len(sample)
    use_shapiro = n < _SHAPIRO_MAX_N
    result = {
        "test": "Shapiro-Wilk" if use_shapiro else "D'Agostino-Pearson",
        "scipy": "scipy.stats.shapiro(data)" if use_shapiro else "scipy.stats.normaltest(data)",
        "n": n,
        "p_value": None,
        "is_normal": True,
        "constant": n > 1 and bool(np.ptp(sample) == 0),
    }
    if result["constant"]:
        result["is_normal"] = False
    elif n >= _MIN_NORMALITY_N:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, p = stats.shapiro(sample) if use_shapiro else stats.normaltest(sample)
        result["p_value"] = float(p)
        result["is_normal"] = bool(p > alpha)
    return result


def _equal_variances(group_data: list) -> bool:
    """Levene's test for homogeneity of variance (True when not rejected/untestable)."""
    if len(group_data) < 2 or any(len(g) < 2 for g in group_data):
        return True
    if all(np.ptp(g) == 0 for g in group_data):
        return True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, p = stats.levene(*group_data)
    return not (np.isfinite(p) and p < _ALPHA)


def _group_comparison(group_labels, group_data: list, paired: bool) -> dict:
    """Recommend a group-comparison test from per-group characteristics."""
    n_groups = len(group_data)
    sizes = [len(g) for g in group_data]
    notes = []

    if paired and len(set(sizes)) > 1:
        notes.append(
            f"paired=True but group sizes differ ({sizes}); paired tests need matched "
            "samples, so independent-samples tests are recommended instead."
        )
        paired = False
    if min(sizes) < 2:
        small = [str(g) for g, n in zip(group_labels, sizes, strict=True) if n < 2]
        notes.append(
            f"Group(s) {', '.join(small)} have fewer than 2 observations; "
            "group comparisons are not meaningful. Collect more data."
        )

    if paired and n_groups == 2:
        # Paired designs: the assumption concerns the within-subject differences.
        checks = [_normality(group_data[1] - group_data[0])]
        what = "paired differences"
    else:
        # Normality is assessed within each group/condition.  Testing the pooled
        # data is wrong: groups with different means make it multimodal.  The
        # per-group level is Bonferroni-adjusted so that normal data is not
        # rejected more often just because there are more groups.
        alpha = _ALPHA / n_groups
        checks = [_normality(g, alpha) for g in group_data]
        what = "each condition" if paired else "each group"
    normal = all(c["is_normal"] for c in checks)
    equal_var = _equal_variances(group_data)
    tested = [_format_p(c["p_value"]) for c in checks if c["p_value"] is not None]
    basis = f"normality of {what}" + (f" ({', '.join(tested)})" if tested else " (n too small)")

    if n_groups == 2:
        if paired and normal:
            rec = {
                "test": "Paired t-test",
                "purpose": "Compare means of 2 paired groups",
                "rationale": f"Paired differences are normally distributed; {basis}",
                "scipy": "scipy.stats.ttest_rel(group1, group2)",
            }
        elif paired:
            rec = {
                "test": "Wilcoxon signed-rank",
                "purpose": "Compare distributions of 2 paired groups",
                "rationale": f"Paired differences are non-normal; {basis}",
                "scipy": "scipy.stats.wilcoxon(group1, group2)",
            }
        elif normal and equal_var:
            rec = {
                "test": "Independent t-test",
                "purpose": "Compare means of 2 groups",
                "rationale": f"Data is normally distributed within groups, 2 groups; {basis}",
                "scipy": "scipy.stats.ttest_ind(group1, group2)",
            }
        elif normal:
            rec = {
                "test": "Welch's t-test",
                "purpose": "Compare means of 2 groups with unequal variances",
                "rationale": f"Normal within groups but variances differ (Levene p<0.05); {basis}",
                "scipy": "scipy.stats.ttest_ind(group1, group2, equal_var=False)",
            }
        else:
            rec = {
                "test": "Mann-Whitney U",
                "purpose": "Compare distributions of 2 groups",
                "rationale": f"Data is non-normal within groups, 2 groups; {basis}",
                "scipy": "scipy.stats.mannwhitneyu(group1, group2)",
            }
    elif paired and normal:
        rec = {
            "test": "Repeated-measures ANOVA",
            "purpose": f"Compare means across {n_groups} paired conditions",
            "rationale": f"Normally distributed repeated measures, 3+ conditions; {basis}",
            "scipy": "statsmodels.stats.anova.AnovaRM(df, depvar, subject, within).fit()",
            "post_hoc": "Paired t-tests with Bonferroni/Holm correction",
        }
    elif paired:
        rec = {
            "test": "Friedman test",
            "purpose": f"Compare distributions across {n_groups} paired conditions",
            "rationale": f"Non-normal repeated measures, 3+ conditions; {basis}",
            "scipy": "scipy.stats.friedmanchisquare(*groups)",
            "post_hoc": "Nemenyi test (scikit-posthocs)",
        }
    elif normal and equal_var:
        rec = {
            "test": "One-way ANOVA",
            "purpose": f"Compare means across {n_groups} groups",
            "rationale": f"Data is normally distributed within groups, 3+ groups; {basis}",
            "scipy": "scipy.stats.f_oneway(*groups)",
            "post_hoc": "Tukey HSD: scipy.stats.tukey_hsd(*groups)",
        }
    elif normal:
        rec = {
            "test": "Welch's ANOVA",
            "purpose": f"Compare means across {n_groups} groups with unequal variances",
            "rationale": f"Normal within groups but variances differ (Levene p<0.05); {basis}",
            "scipy": "statsmodels.stats.oneway.anova_oneway(groups, use_var='unequal')",
            "post_hoc": "Games-Howell test",
        }
    else:
        rec = {
            "test": "Kruskal-Wallis H",
            "purpose": f"Compare distributions across {n_groups} groups",
            "rationale": f"Data is non-normal within groups, 3+ groups; {basis}",
            "scipy": "scipy.stats.kruskal(*groups)",
            "post_hoc": "Dunn's test (scikit-posthocs)",
        }
    labels = [g.item() if hasattr(g, "item") else g for g in group_labels]
    rec["group_sizes"] = dict(zip(labels, sizes, strict=True))
    if notes:
        rec["warning"] = " ".join(notes)
    return rec


def recommend_statistical_tests(
    data: np.ndarray, groups: np.ndarray = None, paired: bool = False
) -> list:
    """Recommend appropriate statistical tests based on data characteristics.

    Parameters
    ----------
    data : np.ndarray
        Data array (1D for single-sample, 2D for multi-variable). Any array-like
        is accepted; NaN/inf values are excluded (and reported in a ``note``).
    groups : np.ndarray, optional
        Group labels for group comparison tests (same length as ``data``).
        Normality is assessed within each group, variance homogeneity with
        Levene's test.
    paired : bool
        Whether groups are paired/matched samples (observations matched by order
        within each group; all groups must have the same size).

    Returns
    -------
    list[dict]
        Recommended tests with rationale and scipy function references.
    """
    data = np.asarray(data, dtype=float)
    recommendations = []

    if data.ndim == 1:
        if groups is not None:
            groups = np.asarray(groups)
            if groups.shape != data.shape:
                raise ValueError(
                    f"groups must be a 1-D array with one label per data value "
                    f"(got shape {groups.shape} for data of shape {data.shape})"
                )
        keep = np.isfinite(data)
        if groups is not None and groups.dtype.kind == "f":
            keep &= ~np.isnan(groups)
        n_dropped = int((~keep).sum())
        if n_dropped:
            data = data[keep]
            groups = groups[keep] if groups is not None else None

        normality = _normality(data)
        is_normal = normality["is_normal"]
        p_normal = normality["p_value"]
        if normality["constant"]:
            result = "zero variance"
            conclusion = "Constant data"
        else:
            result = _format_p(p_normal) if p_normal is not None else "n too small"
            conclusion = "Normal" if is_normal else "Non-normal"
        rec = {
            "test": normality["test"],
            "purpose": "Normality check",
            "result": result,
            "conclusion": conclusion,
            "scipy": normality["scipy"],
            "p_value": p_normal,
            "n": len(data),
        }
        if n_dropped:
            rec["note"] = f"{n_dropped} non-finite value(s) (NaN/inf) excluded"
        if normality["constant"] or len(data) < 2:
            rec["warning"] = (
                "Data has no variability (or fewer than 2 values); "
                "group comparison and correlation tests are undefined."
            )
        recommendations.append(rec)
        degenerate = "warning" in rec

        if groups is not None and not degenerate:
            unique_groups = np.unique(groups)
            group_data = [data[groups == g] for g in unique_groups]
            if len(unique_groups) >= 2:
                recommendations.append(_group_comparison(unique_groups, group_data, paired))

        # Correlation
        if not degenerate:
            recommendations.append(
                {
                    "test": "Pearson r" if is_normal else "Spearman rho",
                    "purpose": "Correlation between variables",
                    "rationale": f"Data is {'' if is_normal else 'non-'}normally distributed",
                    "scipy": "scipy.stats.pearsonr(x, y)"
                    if is_normal
                    else "scipy.stats.spearmanr(x, y)",
                }
            )

    elif data.ndim == 2:
        _, n_features = data.shape
        recommendations.append(
            {
                "test": "Pearson correlation matrix",
                "purpose": f"Pairwise correlations among {n_features} variables",
                "scipy": "np.corrcoef(data, rowvar=False)",
            }
        )
        if n_features <= 20:
            recommendations.append(
                {
                    "test": "Principal Component Analysis",
                    "purpose": "Dimensionality reduction / feature analysis",
                    "scipy": "sklearn.decomposition.PCA(n_components=k).fit(data)",
                }
            )
        if not np.isfinite(data).all():
            recommendations[0]["note"] = (
                "Data contains NaN/inf; drop or impute them first (e.g. pandas.DataFrame.corr)."
            )

    # Effect size
    recommendations.append(
        {
            "test": "Cohen's d",
            "purpose": "Effect size for mean comparisons",
            "formula": "d = (mean1 - mean2) / pooled_std",
            "interpretation": "small=0.2, medium=0.5, large=0.8",
        }
    )

    return recommendations


def recommend_figure_type(
    data: np.ndarray,
    n_variables: int = None,
    data_type: str = "continuous",
    comparison: str = None,
) -> list:
    """Recommend appropriate figure types based on data characteristics.

    Parameters
    ----------
    data : np.ndarray
        The data to visualize.
    n_variables : int, optional
        Number of variables (inferred from data if not provided).
    data_type : str
        "continuous", "categorical", "time_series", "spatial"
    comparison : str, optional
        "groups", "correlation", "distribution", "trend"

    Returns
    -------
    list[dict]
        Recommended figure types with matplotlib code snippets.
    """
    data = np.asarray(data)
    if n_variables is None:
        n_variables = data.shape[1] if data.ndim == 2 else 1

    recommendations = []

    if data_type == "continuous":
        if comparison == "groups" or comparison is None:
            recommendations.append(
                {
                    "type": "Box plot",
                    "best_for": "Comparing distributions across groups",
                    "when": "3-10 groups, showing median/quartiles/outliers",
                    "code": "ax.boxplot(data, patch_artist=True)",
                }
            )
            recommendations.append(
                {
                    "type": "Violin plot",
                    "best_for": "Distribution shape comparison across groups",
                    "when": "When distribution shape matters more than summary stats",
                    "code": "ax.violinplot(data)",
                }
            )
            recommendations.append(
                {
                    "type": "Bar chart with error bars",
                    "best_for": "Comparing means with confidence intervals",
                    "when": "Presenting model performance metrics",
                    "code": "ax.bar(x, means, yerr=stds)",
                }
            )

        if comparison == "correlation" or n_variables >= 2:
            recommendations.append(
                {
                    "type": "Scatter plot",
                    "best_for": "Relationship between two continuous variables",
                    "when": "Checking linearity, outliers, clusters",
                    "code": "ax.scatter(x, y, alpha=0.5)",
                }
            )
            if n_variables > 3:
                recommendations.append(
                    {
                        "type": "Heatmap (correlation matrix)",
                        "best_for": "Pairwise correlations among many variables",
                        "when": "Feature analysis, multicollinearity check",
                        "code": "sns.heatmap(df.corr(), annot=True, cmap='RdBu_r')",
                    }
                )

        if comparison == "trend":
            recommendations.append(
                {
                    "type": "Line plot with markers",
                    "best_for": "Trend of a response versus a continuous parameter",
                    "when": "Parametric studies, convergence, response vs. load/time",
                    "code": "ax.plot(x, y, marker='o')",
                }
            )

        if comparison == "distribution":
            recommendations.append(
                {
                    "type": "Histogram",
                    "best_for": "Single variable distribution",
                    "when": "Checking normality, skewness, modality",
                    "code": "ax.hist(data, bins=50, density=True)",
                }
            )
            recommendations.append(
                {
                    "type": "Q-Q plot",
                    "best_for": "Normality assessment",
                    "when": "Residual analysis, distribution validation",
                    "code": "stats.probplot(data, plot=ax)",
                }
            )

    elif data_type == "time_series":
        recommendations.append(
            {
                "type": "Line plot",
                "best_for": "Temporal trends and patterns",
                "when": "Time history, signal visualization",
                "code": "ax.plot(t, data, linewidth=0.5)",
            }
        )
        recommendations.append(
            {
                "type": "PSD (log-log)",
                "best_for": "Frequency content analysis",
                "when": "Signal processing, spectral characteristics",
                "code": "ax.loglog(f, psd)",
            }
        )

    elif data_type == "spatial":
        recommendations.append(
            {
                "type": "Contour plot",
                "best_for": "2D spatial distribution",
                "when": "Pressure coefficients, temperature fields",
                "code": "ax.contourf(X, Y, Z, levels=20, cmap='RdBu_r')",
            }
        )
        recommendations.append(
            {
                "type": "Polar plot",
                "best_for": "Directional dependence (wind direction, etc.)",
                "when": "Wind direction analysis, angular data",
                "code": "ax.plot(theta, r) # subplot_kw={'projection': 'polar'}",
            }
        )

    elif data_type == "categorical":
        recommendations.append(
            {
                "type": "Confusion matrix",
                "best_for": "Classification performance visualization",
                "when": "Multi-class classification results",
                "code": "ax.imshow(cm, cmap='Blues')",
            }
        )
        recommendations.append(
            {
                "type": "Grouped bar chart",
                "best_for": "Per-class accuracy comparison",
                "when": "Comparing multiple models across categories",
                "code": "ax.bar(x + offset, values, width)",
            }
        )

    # Always recommend for ML papers
    if n_variables >= 5:
        recommendations.append(
            {
                "type": "Feature importance (horizontal bar)",
                "best_for": "Ranking feature contributions",
                "when": "Random Forest / Gradient Boosting feature analysis",
                "code": "ax.barh(features, importances)",
            }
        )

    return recommendations
