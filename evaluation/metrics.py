"""
Success criteria metrics for Phase 1 validation.

Five quantitative criteria that must all be met for Phase 1 to be considered
successful. Each function returns the metric value and whether it passes.

Criterion 1: Profile Reconstruction R^2 >= 0.7
Criterion 2: Geometric Consistency (per-layer Spearman rho > 0.5, mean > 0.65)
Criterion 3: Within-Span Similarity Elevation (cluster mean > pop mean + 1 std)
Criterion 4: Circuit Diversity (spans cover >= 60% of layer range)
Criterion 5: Class Purity Distribution (bimodal: both <0.3 and >0.7 circuits)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def profile_reconstruction_r2(
    predicted: np.ndarray, true: np.ndarray
) -> dict:
    """
    Criterion 1: Profile Reconstruction Accuracy.

    Args:
        predicted: [N_pairs, L] predicted per-layer similarities
        true:      [N_pairs, L] ground-truth per-layer similarities

    Returns:
        dict with 'r2' (float) and 'passes' (bool, R^2 >= 0.7)
    """
    ss_res = np.sum((predicted - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    return {"r2": float(r2), "passes": r2 >= 0.7}


def geometric_consistency(
    z_sims: np.ndarray, true_sims: np.ndarray, n_layers: int
) -> dict:
    """
    Criterion 2: Geometric Consistency.

    Computes per-layer Spearman correlation between true profile similarity
    and z-space cosine similarity.

    Args:
        z_sims:    [N_pairs, L] pairwise z-space cosine similarities
        true_sims: [N_pairs, L] pairwise true profile similarities
        n_layers:  number of layers L

    Returns:
        dict with 'per_layer_rho' (list), 'mean_rho' (float), 'passes' (bool)
    """
    per_layer_rho = []
    for l in range(n_layers):
        rho, _ = spearmanr(z_sims[:, l], true_sims[:, l])
        per_layer_rho.append(float(rho) if not np.isnan(rho) else 0.0)

    mean_rho = float(np.mean(per_layer_rho))
    all_above_05 = all(r > 0.5 for r in per_layer_rho)
    passes = all_above_05 and mean_rho > 0.65

    return {
        "per_layer_rho": per_layer_rho,
        "mean_rho": mean_rho,
        "passes": passes,
    }


def within_span_elevation(
    cluster_similarities: np.ndarray, population_similarities: np.ndarray
) -> dict:
    """
    Criterion 3: Within-Span Similarity Elevation.

    The cluster mean within-span similarity must exceed the population mean
    by at least one population standard deviation.

    Args:
        cluster_similarities:    [N_cluster] mean within-span sims for cluster pairs
        population_similarities: [N_total] mean within-span sims for all pairs

    Returns:
        dict with 'cluster_mean', 'pop_mean', 'pop_std', 'elevation',
        'passes' (bool)
    """
    cluster_mean = float(np.mean(cluster_similarities))
    pop_mean = float(np.mean(population_similarities))
    pop_std = float(np.std(population_similarities))

    elevation = (cluster_mean - pop_mean) / max(pop_std, 1e-8)
    passes = cluster_mean > pop_mean + pop_std

    return {
        "cluster_mean": cluster_mean,
        "pop_mean": pop_mean,
        "pop_std": pop_std,
        "elevation_sigma": elevation,
        "passes": passes,
    }


def circuit_diversity(
    circuit_spans: list[tuple[int, int]], total_layers: int
) -> dict:
    """
    Criterion 4: Circuit Diversity.

    The discovered circuits must collectively cover at least 60% of the
    total layer range [1, L].

    Args:
        circuit_spans: list of (l_start, l_end) for each canonical circuit
        total_layers:  total number of layers L

    Returns:
        dict with 'coverage' (float, 0-1), 'covered_layers' (set),
        'passes' (bool, coverage >= 0.6)
    """
    covered = set()
    for l_start, l_end in circuit_spans:
        for l in range(l_start, l_end + 1):
            covered.add(l)

    coverage = len(covered) / total_layers if total_layers > 0 else 0.0
    return {
        "coverage": float(coverage),
        "covered_layers": covered,
        "n_circuits": len(circuit_spans),
        "passes": coverage >= 0.6,
    }


def class_purity_distribution(
    purities: list[float],
) -> dict:
    """
    Criterion 5: Class Purity Distribution is Bimodal or Mixed.

    Among all canonical circuits, the purity distribution must contain
    both class-agnostic (purity < 0.3) and class-specific (purity > 0.7)
    circuits.

    Args:
        purities: list of class purity scores for canonical circuits

    Returns:
        dict with 'n_agnostic' (<0.3), 'n_specific' (>0.7), 'n_total',
        'passes' (bool, both > 0)
    """
    purities_arr = np.array(purities)
    n_agnostic = int(np.sum(purities_arr < 0.3))
    n_specific = int(np.sum(purities_arr > 0.7))
    n_middle = int(np.sum((purities_arr >= 0.3) & (purities_arr <= 0.7)))

    passes = n_agnostic > 0 and n_specific > 0

    return {
        "n_agnostic": n_agnostic,
        "n_specific": n_specific,
        "n_middle": n_middle,
        "n_total": len(purities),
        "passes": passes,
    }
