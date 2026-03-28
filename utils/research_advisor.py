"""Research methodology advisors: statistical tests and figure type recommendations."""

import numpy as np
from scipy import stats


def recommend_statistical_tests(data: np.ndarray, groups: np.ndarray = None, paired: bool = False) -> list:
    """Recommend appropriate statistical tests based on data characteristics.

    Parameters
    ----------
    data : np.ndarray
        Data array (1D for single-sample, 2D for multi-variable).
    groups : np.ndarray, optional
        Group labels for group comparison tests.
    paired : bool
        Whether groups are paired/matched samples.

    Returns
    -------
    list[dict]
        Recommended tests with rationale and scipy function references.
    """
    recommendations = []

    if data.ndim == 1:
        n = len(data)

        # Normality test
        if n >= 8:
            _, p_normal = stats.shapiro(data) if n < 5000 else stats.normaltest(data)
            is_normal = p_normal > 0.05
        else:
            is_normal = True
            p_normal = None

        recommendations.append({
            "test": "Shapiro-Wilk" if n < 5000 else "D'Agostino-Pearson",
            "purpose": "Normality check",
            "result": f"p={p_normal:.4f}" if p_normal else "n too small",
            "conclusion": "Normal" if is_normal else "Non-normal",
            "scipy": "scipy.stats.shapiro(data)",
        })

        if groups is not None:
            unique_groups = np.unique(groups)
            n_groups = len(unique_groups)
            group_data = [data[groups == g] for g in unique_groups]

            if n_groups == 2:
                if is_normal:
                    test_name = "Paired t-test" if paired else "Independent t-test"
                    scipy_func = "scipy.stats.ttest_rel" if paired else "scipy.stats.ttest_ind"
                    recommendations.append({
                        "test": test_name,
                        "purpose": "Compare means of 2 groups",
                        "rationale": "Data is normally distributed, 2 groups",
                        "scipy": f"{scipy_func}(group1, group2)",
                    })
                else:
                    test_name = "Wilcoxon signed-rank" if paired else "Mann-Whitney U"
                    scipy_func = "scipy.stats.wilcoxon" if paired else "scipy.stats.mannwhitneyu"
                    recommendations.append({
                        "test": test_name,
                        "purpose": "Compare distributions of 2 groups",
                        "rationale": "Data is non-normal, 2 groups",
                        "scipy": f"{scipy_func}(group1, group2)",
                    })
            elif n_groups > 2:
                if is_normal:
                    recommendations.append({
                        "test": "One-way ANOVA",
                        "purpose": f"Compare means across {n_groups} groups",
                        "rationale": "Data is normally distributed, 3+ groups",
                        "scipy": "scipy.stats.f_oneway(*groups)",
                        "post_hoc": "Tukey HSD: scipy.stats.tukey_hsd(*groups)",
                    })
                else:
                    recommendations.append({
                        "test": "Kruskal-Wallis H",
                        "purpose": f"Compare distributions across {n_groups} groups",
                        "rationale": "Data is non-normal, 3+ groups",
                        "scipy": "scipy.stats.kruskal(*groups)",
                        "post_hoc": "Dunn's test (scikit-posthocs)",
                    })

        # Correlation
        recommendations.append({
            "test": "Pearson r" if is_normal else "Spearman rho",
            "purpose": "Correlation between variables",
            "rationale": f"Data is {'normally' if is_normal else 'non-normally'} distributed",
            "scipy": "scipy.stats.pearsonr(x, y)" if is_normal else "scipy.stats.spearmanr(x, y)",
        })

    elif data.ndim == 2:
        n_samples, n_features = data.shape
        recommendations.append({
            "test": "Pearson correlation matrix",
            "purpose": f"Pairwise correlations among {n_features} variables",
            "scipy": "np.corrcoef(data, rowvar=False)",
        })
        if n_features <= 20:
            recommendations.append({
                "test": "Principal Component Analysis",
                "purpose": "Dimensionality reduction / feature analysis",
                "scipy": "sklearn.decomposition.PCA(n_components=k).fit(data)",
            })

    # Effect size
    recommendations.append({
        "test": "Cohen's d",
        "purpose": "Effect size for mean comparisons",
        "formula": "d = (mean1 - mean2) / pooled_std",
        "interpretation": "small=0.2, medium=0.5, large=0.8",
    })

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
    if n_variables is None:
        n_variables = data.shape[1] if data.ndim == 2 else 1

    recommendations = []

    if data_type == "continuous":
        if comparison == "groups" or comparison is None:
            recommendations.append({
                "type": "Box plot",
                "best_for": "Comparing distributions across groups",
                "when": "3-10 groups, showing median/quartiles/outliers",
                "code": "ax.boxplot(data, patch_artist=True)",
            })
            recommendations.append({
                "type": "Violin plot",
                "best_for": "Distribution shape comparison across groups",
                "when": "When distribution shape matters more than summary stats",
                "code": "ax.violinplot(data)",
            })
            recommendations.append({
                "type": "Bar chart with error bars",
                "best_for": "Comparing means with confidence intervals",
                "when": "Presenting model performance metrics",
                "code": "ax.bar(x, means, yerr=stds)",
            })

        if comparison == "correlation" or n_variables >= 2:
            recommendations.append({
                "type": "Scatter plot",
                "best_for": "Relationship between two continuous variables",
                "when": "Checking linearity, outliers, clusters",
                "code": "ax.scatter(x, y, alpha=0.5)",
            })
            if n_variables > 3:
                recommendations.append({
                    "type": "Heatmap (correlation matrix)",
                    "best_for": "Pairwise correlations among many variables",
                    "when": "Feature analysis, multicollinearity check",
                    "code": "sns.heatmap(df.corr(), annot=True, cmap='RdBu_r')",
                })

        if comparison == "distribution":
            recommendations.append({
                "type": "Histogram",
                "best_for": "Single variable distribution",
                "when": "Checking normality, skewness, modality",
                "code": "ax.hist(data, bins=50, density=True)",
            })
            recommendations.append({
                "type": "Q-Q plot",
                "best_for": "Normality assessment",
                "when": "Residual analysis, distribution validation",
                "code": "stats.probplot(data, plot=ax)",
            })

    elif data_type == "time_series":
        recommendations.append({
            "type": "Line plot",
            "best_for": "Temporal trends and patterns",
            "when": "Time history, signal visualization",
            "code": "ax.plot(t, data, linewidth=0.5)",
        })
        recommendations.append({
            "type": "PSD (log-log)",
            "best_for": "Frequency content analysis",
            "when": "Signal processing, spectral characteristics",
            "code": "ax.loglog(f, psd)",
        })

    elif data_type == "spatial":
        recommendations.append({
            "type": "Contour plot",
            "best_for": "2D spatial distribution",
            "when": "Pressure coefficients, temperature fields",
            "code": "ax.contourf(X, Y, Z, levels=20, cmap='RdBu_r')",
        })
        recommendations.append({
            "type": "Polar plot",
            "best_for": "Directional dependence (wind direction, etc.)",
            "when": "Wind direction analysis, angular data",
            "code": "ax.plot(theta, r) # subplot_kw={'projection': 'polar'}",
        })

    elif data_type == "categorical":
        recommendations.append({
            "type": "Confusion matrix",
            "best_for": "Classification performance visualization",
            "when": "Multi-class classification results",
            "code": "ax.imshow(cm, cmap='Blues')",
        })
        recommendations.append({
            "type": "Grouped bar chart",
            "best_for": "Per-class accuracy comparison",
            "when": "Comparing multiple models across categories",
            "code": "ax.bar(x + offset, values, width)",
        })

    # Always recommend for ML papers
    if n_variables >= 5:
        recommendations.append({
            "type": "Feature importance (horizontal bar)",
            "best_for": "Ranking feature contributions",
            "when": "Random Forest / Gradient Boosting feature analysis",
            "code": "ax.barh(features, importances)",
        })

    return recommendations
