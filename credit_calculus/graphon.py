"""Graphon topologies and Watts-Strogatz communication neighborhoods."""

from __future__ import annotations

import numpy as np
import networkx as nx

from .config import GraphonTopology, VPPConfig, DEFAULT_CONFIG


def build_watts_strogatz(n: int, config: VPPConfig = DEFAULT_CONFIG) -> nx.Graph:
    k = min(config.ws_k, n - 1)
    if k < 2:
        return nx.complete_graph(n) if n > 1 else nx.Graph()
    return nx.watts_strogatz_graph(n, k, config.ws_p, seed=42)


def latent_positions(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=n)


def graphon_weight(
    alpha: float,
    beta: float,
    topology: GraphonTopology,
    subdivisions: int = 4,
    eta: float = 0.5,
    theta1: float = 1.0,
    theta2: float = 0.1,
) -> float:
    if topology == GraphonTopology.FLAT:
        return 1.0
    if topology == GraphonTopology.HIERARCHICAL:
        return 1.0 if int(alpha * subdivisions) == int(beta * subdivisions) else 0.0
    if topology == GraphonTopology.SCALE_FREE:
        return float((max(alpha, 1e-3) * max(beta, 1e-3)) ** (-eta))
    if topology == GraphonTopology.NESTED_GRID:
        same = int(alpha * subdivisions) == int(beta * subdivisions)
        return theta1 if same else theta2
    return 1.0


def weight_matrix(
    alphas: np.ndarray,
    topology: GraphonTopology,
) -> np.ndarray:
    n = len(alphas)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            W[i, j] = graphon_weight(float(alphas[i]), float(alphas[j]), topology)
    return W


def latency_rbf_weights(
    W: np.ndarray,
    delays_ms: np.ndarray,
    decay: float = DEFAULT_CONFIG.graphon_decay,
) -> np.ndarray:
    """Latency-discounted RBF kernel (Section 10)."""
    kernel = np.exp(-decay * delays_ms / 100.0)
    return W * kernel


def neighborhood_indices(
    graph: nx.Graph,
    i: int,
    max_size: int = DEFAULT_CONFIG.neighborhood_size,
) -> list[int]:
    """Active causal neighborhood Ki: graph neighbors + self, capped."""
    if i not in graph:
        return [i]
    nbrs = list(graph.neighbors(i))
    # Expand to 2-hop if needed to reach target size
    expanded = set(nbrs)
    for nb in nbrs:
        expanded.update(graph.neighbors(nb))
    expanded.discard(i)
    ordered = sorted(expanded)
    if len(ordered) > max_size - 1:
        ordered = ordered[: max_size - 1]
    return [i] + ordered


def delay_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """Round-trip delays 10–120 ms between agent pairs."""
    base = rng.uniform(10.0, 120.0, size=(n, n))
    D = 0.5 * (base + base.T)
    np.fill_diagonal(D, 0.0)
    return D
