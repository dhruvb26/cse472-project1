import logging
import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def load_or_build_graph(
    gexf_path: str = "users_network.gexf",
) -> nx.Graph | nx.DiGraph | None:
    """
    Load a graph from a GEXF file. If it does not exist (or rebuild=True),
    build it from users.json using users.create_user_network_graph and save it.
    """
    try:
        if os.path.exists(gexf_path):
            return nx.read_gexf(gexf_path)
    except Exception:
        logger.exception("Error in load_or_build_graph: ")
        raise


def compute_degree_counts(graph: nx.Graph | nx.DiGraph) -> dict[int, int]:
    """
    Compute degree counts for the graph (total degree for directed graphs).
    Returns a mapping: degree -> count of nodes with that degree.
    """
    degrees = [int(graph.degree[n]) for n in graph.nodes()]
    counts = Counter(degrees)
    # Ensure standard dict with int keys
    return {int(k): int(v) for k, v in counts.items()}


def plot_degree_histogram(
    degree_counts: dict[int, int],
    title: str = "Degree Distribution (Users Network)",
    out_path: str = "users_degree_dist.png",
) -> str:
    """
    Plot and save a histogram-style bar chart for discrete degree distribution.
    Returns the saved file path.
    """
    if not degree_counts:
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "No nodes in graph", ha="center", va="center", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        return out_path

    xs = sorted(degree_counts.keys())
    ys = [degree_counts[x] for x in xs]
    total_nodes = sum(ys)

    plt.figure(figsize=(12, 8))
    plt.rcParams.update(
        {"font.size": 12, "font.family": "sans-serif", "font.sans-serif": ["Arial"]}
    )

    plt.bar(
        xs,
        ys,
        width=0.8,
        align="center",
        color="#ff6f61",
        alpha=1.0,
        edgecolor="black",
        linewidth=0.5,
    )

    plt.xlabel("Degree (k)", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Nodes", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)

    plt.grid(axis="y", linestyle="--", alpha=0.7, color="gray", linewidth=0.5)
    plt.grid(axis="x", linestyle=":", alpha=0.3, color="gray", linewidth=0.3)

    formula_text = (
        r"$P(k) = \frac{N_k}{N}$"
        + "\n"
        + r"where $N_k$ = number of nodes with degree $k$"
        + "\n"
        + r"and $N$ = total number of nodes"
    )

    plt.text(
        0.98,
        0.95,
        formula_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    max_degree = max(xs)
    avg_degree = sum(k * count for k, count in degree_counts.items()) / total_nodes
    stats_text = f"Total nodes: {total_nodes}\nMax degree: {max_degree}\nAvg degree: {avg_degree:.2f}"

    plt.text(
        0.98,
        0.80,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="png",
    )

    plt.close()
    return out_path


def plot_degree_distribution(
    degree_counts: dict[int, int],
    title: str = "Degree Distribution P(k) (Users Network)",
    out_path: str = "users_degree_dist_P(k).png",
    log_scale: bool = True,
) -> str:
    """
    Plot the degree distribution as probabilities P(k) = N_k/N.
    Similar to the Facebook example with log-log scale.
    Returns the saved file path.
    """
    if not degree_counts:
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "No nodes in graph", ha="center", va="center", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        return out_path

    total_nodes = sum(degree_counts.values())
    degrees = sorted(degree_counts.keys())
    probabilities = [degree_counts[k] / total_nodes for k in degrees]

    plt.figure(figsize=(12, 8))
    plt.rcParams.update(
        {"font.size": 12, "font.family": "sans-serif", "font.sans-serif": ["Arial"]}
    )

    plt.plot(
        degrees,
        probabilities,
        "o-",
        color="#ff9800",
        linewidth=2,
        markersize=4,
        markerfacecolor="#ff7043",
        markeredgecolor="white",
        markeredgewidth=0.5,
        alpha=0.8,
    )

    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Degree", fontsize=14, fontweight="bold")
        plt.ylabel("Fraction", fontsize=14, fontweight="bold")
    else:
        plt.xlabel("Degree (k)", fontsize=14, fontweight="bold")
        plt.ylabel("P(k)", fontsize=14, fontweight="bold")

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="png",
    )

    plt.close()
    return out_path


def compute_betweenness_centrality(
    graph: nx.Graph | nx.DiGraph, normalized: bool = True
) -> dict[str, float]:
    """
    Compute betweenness centrality for all nodes in the graph.

    Betweenness centrality C_b(v_i) = ∑ (σ_st(v_i) / σ_st) for s≠t≠v_i
    where σ_st is number of shortest paths from s to t,
    and σ_st(v_i) is number of shortest paths from s to t passing through v_i.

    If normalized=True, divides by (n-1)(n-2) for undirected graphs.
    """
    return nx.betweenness_centrality(graph, normalized=normalized)


def compute_closeness_centrality(graph: nx.Graph | nx.DiGraph) -> dict[str, float]:
    """
    Compute closeness centrality for all nodes in the graph.

    Closeness centrality C_c(v_i) = (n-1) / ∑ d(v_i, v_j) for j≠i
    where d(v_i, v_j) is the shortest path distance between nodes v_i and v_j,
    and n is the number of nodes in the graph.

    The closeness centrality is automatically normalized by NetworkX.
    """
    return nx.closeness_centrality(graph)


def plot_degree_vs_betweenness(
    graph: nx.Graph | nx.DiGraph,
    betweenness_values: dict[str, float],
    title: str = "Degree vs Betweenness Centrality",
    out_path: str = "users_degree_vs_betweenness.png",
) -> str:
    """
    Plot scatter plot of degree vs betweenness centrality to show relationship.
    """
    if not betweenness_values:
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "No nodes in graph", ha="center", va="center", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        return out_path

    degrees = [graph.degree[node] for node in betweenness_values.keys()]
    betweenness = list(betweenness_values.values())

    plt.figure(figsize=(12, 8))
    plt.rcParams.update(
        {"font.size": 12, "font.family": "sans-serif", "font.sans-serif": ["Arial"]}
    )

    plt.scatter(
        degrees,
        betweenness,
        alpha=0.6,
        color="#ff5722",
        s=20,
        edgecolors="white",
        linewidth=0.5,
    )

    plt.xlabel("Degree", fontsize=14, fontweight="bold")
    plt.ylabel("Betweenness Centrality", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)

    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Add correlation info
    correlation = np.corrcoef(degrees, betweenness)[0, 1]
    corr_text = f"Correlation: {correlation:.3f}"

    plt.text(
        0.02,
        0.95,
        corr_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="png",
    )

    plt.close()
    return out_path


def plot_degree_vs_closeness(
    graph: nx.Graph | nx.DiGraph,
    closeness_values: dict[str, float],
    title: str = "Degree vs Closeness Centrality",
    out_path: str = "users_degree_vs_closeness.png",
) -> str:
    """
    Plot scatter plot of degree vs closeness centrality to show relationship.
    """
    if not closeness_values:
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "No nodes in graph", ha="center", va="center", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        return out_path

    degrees = [graph.degree[node] for node in closeness_values.keys()]
    closeness = list(closeness_values.values())

    plt.figure(figsize=(12, 8))
    plt.rcParams.update(
        {"font.size": 12, "font.family": "sans-serif", "font.sans-serif": ["Arial"]}
    )

    plt.scatter(
        degrees,
        closeness,
        alpha=0.6,
        color="#d2691e",
        s=20,
        edgecolors="white",
        linewidth=0.5,
    )

    plt.xlabel("Degree", fontsize=14, fontweight="bold")
    plt.ylabel("Closeness Centrality", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)

    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Add correlation info
    correlation = np.corrcoef(degrees, closeness)[0, 1]
    corr_text = f"Correlation: {correlation:.3f}"

    plt.text(
        0.02,
        0.95,
        corr_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="png",
    )

    plt.close()
    return out_path


def calculate_local_level_friends(graph: nx.Graph | nx.DiGraph) -> dict[str, float]:
    """
    Calculate the average number of friends for each individual node
    by considering only its immediate connections (1-hop friends).
    This local average provides insight into the immediate social circle of each node.

    Returns a dictionary mapping node -> average degree of its neighbors.
    """
    local_averages = {}

    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if neighbors:
            # Calculate average degree of the node's neighbors
            neighbor_degrees = [graph.degree[neighbor] for neighbor in neighbors]
            local_avg = sum(neighbor_degrees) / len(neighbor_degrees)
            local_averages[node] = local_avg
        else:
            # Node has no neighbors
            local_averages[node] = 0.0

    return local_averages


def calculate_global_level_friends(graph: nx.Graph | nx.DiGraph) -> float:
    """
    Calculate the overall average number of friends across the entire network.
    This global average gives an overview of the general connectivity and density
    of the network as a whole.

    Returns the global average degree.
    """
    if graph.number_of_nodes() == 0:
        return 0.0

    total_degree = sum(graph.degree[node] for node in graph.nodes())
    global_avg = total_degree / graph.number_of_nodes()

    return global_avg


def plot_local_level_friends_distribution(
    local_averages: dict[str, float],
    title: str = "Local Level Friends Distribution",
    out_path: str = "users_local_friends_dist.png",
) -> str:
    """
    Plot the distribution of local level friends (average degrees of neighbors).
    Shows how the local social circle sizes are distributed across the network.
    Returns the saved file path.
    """
    if not local_averages:
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "No nodes in graph", ha="center", va="center", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        return out_path

    local_values = list(local_averages.values())

    plt.figure(figsize=(12, 8))
    plt.rcParams.update(
        {"font.size": 12, "font.family": "sans-serif", "font.sans-serif": ["Arial"]}
    )

    n_bins = min(30, max(10, int(np.sqrt(len(local_values)))))
    plt.hist(
        local_values,
        bins=n_bins,
        alpha=0.7,
        color="#d2691e",
        edgecolor="black",
        linewidth=0.5,
        density=False,
    )

    plt.xlabel("Local Average (Friends of Friends)", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Nodes", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)

    plt.grid(axis="y", linestyle="--", alpha=0.7, color="gray", linewidth=0.5)
    plt.grid(axis="x", linestyle=":", alpha=0.3, color="gray", linewidth=0.3)

    mean_local = np.mean(local_values)
    std_local = np.std(local_values)
    median_local = np.median(local_values)

    stats_text = (
        f"Mean: {mean_local:.2f}\n"
        f"Std: {std_local:.2f}\n"
        f"Median: {median_local:.2f}\n"
        f"Total nodes: {len(local_values)}"
    )

    plt.text(
        0.98,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        format="png",
    )

    plt.close()
    return out_path


def main() -> None:
    graph = load_or_build_graph()
    if graph is None:
        logger.error("Main execution failed: Graph not found")
        return

    # Degree analysis
    # degree_counts = compute_degree_counts(graph)
    # plot_degree_histogram(degree_counts)
    # plot_degree_distribution(degree_counts)

    # # Centrality analysis
    # betweenness_values = compute_betweenness_centrality(graph, normalized=True)
    # plot_degree_vs_betweenness(graph, betweenness_values)

    # closeness_values = compute_closeness_centrality(graph)
    # plot_degree_vs_closeness(graph, closeness_values)

    # Local Level (1-hop Friends)
    local_averages = calculate_local_level_friends(graph)
    plot_local_level_friends_distribution(local_averages)
    # Global Level (Entire Network)
    global_average = calculate_global_level_friends(graph)
    print(f"\nGlobal Level (Entire Network): {global_average:.2f}")


if __name__ == "__main__":
    main()
