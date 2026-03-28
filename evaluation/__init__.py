from evaluation.circuit_analysis import CircuitAnalyzer
from evaluation.circuit_viz import plot_per_layer_umap, plot_profile_heatmap, plot_span_coverage
from evaluation.discovery import SpanCentricDiscovery
from evaluation.metrics import (
    profile_reconstruction_r2,
    geometric_consistency,
    within_span_elevation,
    circuit_diversity,
    class_purity_distribution,
)
