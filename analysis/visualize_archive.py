import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.lines import Line2D
from utils.generation_records import node_status, write_json

from utils.gl_utils import (
    load_archive_data,
    get_saved_score,
    get_parent_genid,
)
from utils.domain_utils import get_domain_splits, can_domain_ensembled


def _build_archive_graph(archive_data, output_dir, trunc_its=-1):
    """Keep every node, original ID, and parent edge in archive order."""
    G = nx.DiGraph()
    ordered = list(dict.fromkeys(genid for entry in archive_data for genid in entry["archive"]))
    if trunc_its > 0:
        ordered = ordered[:trunc_its]
    allowed = {str(genid) for genid in ordered}
    for genid in ordered:
        node = str(genid)
        metadata_path = os.path.join(output_dir, f"gen_{genid}", "metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as stream:
                metadata = json.load(stream)
        G.add_node(node, status=node_status(metadata),
                   failure_reason=metadata.get("failure_reason") or metadata.get("evaluation_error"),
                   valid_parent=metadata.get("valid_parent"))
        parent = get_parent_genid(output_dir, genid)
        if parent is not None and str(parent) != node and (trunc_its <= 0 or str(parent) in allowed):
            G.add_edge(str(parent), node)
    return G


def build_graph_single_domain(domain, archive_data, output_dir, trunc_its=-1, split="train", type="agent"):
    G = _build_archive_graph(archive_data, output_dir, trunc_its)
    score_map = {node: get_saved_score(domain, output_dir, node, split=split, type=type)
                 for node in G.nodes}
    return G, score_map


def build_graph_together(domains, archive_data, output_dir, trunc_its=-1, split="train", type="agent"):
    G = _build_archive_graph(archive_data, output_dir, trunc_its)
    score_map = {}
    for node in G.nodes:
        scores = [get_saved_score(domain, output_dir, node, split=split, type=type)
                  for domain in domains]
        score_map[node] = float(np.mean(scores)) if scores and all(s is not None for s in scores) else None
    return G, score_map


def _archive_layout(G):
    try:
        return graphviz_layout(G, prog="dot")
    except (ImportError, OSError):
        # A deterministic layered layout keeps plotting usable without Graphviz.
        depth = {}
        for node in nx.topological_sort(G):
            depth[node] = max((depth[parent] + 1 for parent in G.predecessors(node)), default=0)
        positions = {}
        for level in sorted(set(depth.values())):
            layer = [node for node in G.nodes if depth[node] == level]
            for index, node in enumerate(layer):
                positions[node] = (index - (len(layer) - 1) / 2, -level)
        return positions


def visualize_graph(G, score_map, output_dir, name_suffix, split="train", type="agent", plot_borders=False, save_svg=False):
    if not G:
        print("No nodes to visualize.")
        return
    pos = _archive_layout(G)
    valid_scores = {node: score for node, score in score_map.items()
                    if score is not None and np.isfinite(score)}
    best = max(valid_scores, key=valid_scores.get) if valid_scores else None
    minimum, maximum = (min(valid_scores.values()), max(valid_scores.values())) if valid_scores else (0, 1)
    if minimum == maximum:
        minimum, maximum = min(0, minimum), max(1, maximum)
    norm = plt.Normalize(minimum, maximum)
    cmap = plt.get_cmap("viridis")
    styles = {
        "completed": ("o", "#238443", "Completed"),
        "failed": ("X", "#c62828", "Failed"),
        "screening_rejected": ("s", "#d98c00", "Screening rejected"),
        "no_changes": ("h", "#666666", "No changes"),
        "running": ("^", "#2879b9", "Running"),
        "not_evaluated": ("h", "#666666", "Not evaluated"),
        "unknown": ("o", "#666666", "Legacy / unknown"),
    }
    depths = {}
    for node in nx.topological_sort(G):
        depths[node] = max((depths[parent] + 1 for parent in G.predecessors(node)), default=0)
    layers = [sum(depth == level for depth in depths.values()) for level in set(depths.values())]
    fig, ax = plt.subplots(figsize=(max(12, max(layers) * 1.5 + 3),
                                    max(7, (max(depths.values()) + 1) * 1.1 + 2)))
    labels, used = {}, set()
    for node in G.nodes:
        status = G.nodes[node].get("status", "unknown")
        if status not in styles:
            status = "unknown"
        used.add(status)
        shape, border, text = styles[status]
        score = valid_scores.get(node)
        fill = cmap(norm(score)) if score is not None else "#eeeeee"
        if node == best:
            shape = "D"
        if plot_borders and G.nodes[node].get("valid_parent") is False and status == "completed":
            border = "#c62828"
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=[fill],
                               edgecolors=[border], linewidths=2, node_shape=shape,
                               node_size=1450, ax=ax)
        labels[node] = f"#{node}\n{score:.3f}" if score is not None else f"#{node}\nN/A"
        if status != "completed":
            labels[node] += "\n" + ("Rejected" if status == "screening_rejected" else text)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=16, node_size=1450)
    for node, label in labels.items():
        score = valid_scores.get(node)
        fill = cmap(norm(score)) if score is not None else (1, 1, 1, 1)
        luminance = sum(weight * channel for weight, channel in zip((0.2126, 0.7152, 0.0722), fill))
        nx.draw_networkx_labels(G, pos, labels={node: label}, ax=ax, font_size=7,
                                font_color="white" if luminance < 0.4 else "black")
    handles = [Line2D([], [], marker=styles[state][0], color="none", markerfacecolor="#eeeeee",
                      markeredgecolor=styles[state][1], markersize=10, label=styles[state][2])
               for state in styles if state in used]
    if best is not None:
        handles.append(Line2D([], [], marker="D", color="none", markerfacecolor="#eeeeee",
                              markeredgecolor="black", markersize=10, label="Best measured score"))
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    if valid_scores:
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     label="Full validation accuracy" if split == "val" else "Score")
    ax.set_title(f"Agent evolution - {name_suffix} ({split})")
    ax.axis("off")
    fig.tight_layout()
    base = os.path.join(output_dir, f"archive_graph_{name_suffix}_{split}_{type}")
    fig.savefig(base + ".png", dpi=200)
    if save_svg:
        fig.savefig(base + ".svg", transparent=True)
    write_json(base + ".json", dict(
        split=split, nodes=[dict(id=node, score=score_map.get(node), **G.nodes[node]) for node in G.nodes],
        edges=[dict(parent=parent, child=child) for parent, child in G.edges]))
    plt.close(fig)
    print(f"Graph saved to {base}.png")


def visualize_archive_single(domain, exp_dir, trunc_its=-1, split="train", type="agent", plot_borders=False, save_svg=False):
    exp_dir = os.path.normpath(exp_dir)
    archive_path = os.path.join(exp_dir, "archive.jsonl")
    archive_data = load_archive_data(archive_path, last_only=False)

    G, score_map = build_graph_single_domain(domain, archive_data, exp_dir, trunc_its=trunc_its, split=split, type=type)
    visualize_graph(G, score_map, exp_dir, name_suffix=domain, split=split, type=type, plot_borders=plot_borders, save_svg=save_svg)


def visualize_archive_together(domains, exp_dir, trunc_its=-1, split="train", type="agent", plot_borders=False, save_svg=False):
    exp_dir = os.path.normpath(exp_dir)
    archive_path = os.path.join(exp_dir, "archive.jsonl")
    archive_data = load_archive_data(archive_path, last_only=False)

    G, score_map = build_graph_together(domains, archive_data, exp_dir, trunc_its=trunc_its, split=split, type=type)
    name_suffix = "together_" + "_".join(domains)
    visualize_graph(G, score_map, exp_dir, name_suffix=name_suffix, split=split, type=type, plot_borders=plot_borders, save_svg=save_svg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",  # one or more domains
        required=True,
        help="One or more domains to evaluate (must be from the allowed list)",
    )
    parser.add_argument("--path", type=str, default="./outputs/generate_20250617_080733_189684", help="Path to the run.")
    parser.add_argument("--trunc_its", type=int, default=-1, help="Truncate iterations to this number, -1 for no truncation")
    parser.add_argument(
        "--together",
        action="store_true",
        help="If set, plot all specified domains together on a single graph, aggregating scores per node."
    )
    parser.add_argument("--plot_borders", action="store_true", help="If set, plot node borders based on valid_parent metadata.")
    parser.add_argument("--check_ensemble", action="store_true", help="If set, check if domains support ensembling.")
    parser.add_argument("--svg", action="store_true", help="If set, save the figure as SVG format.")
    args = parser.parse_args()

    exp_dir = args.path
    domains = args.domains

    if args.together:
        # Determine common splits across the chosen domains
        domain_splits = [set(get_domain_splits(d)) for d in domains]
        common_splits = sorted(list(set.intersection(*domain_splits))) if domain_splits else []

        if not common_splits:
            print("No common splits across the selected domains. Nothing to plot.")
            raise SystemExit(0)

        # Only use ensemble types if ALL domains support ensembling
        ensemble_domain_all = args.check_ensemble and all(can_domain_ensembled(d) for d in domains)
        score_types = ["agent", "ensemble", "max"] if ensemble_domain_all else ["agent"]

        for split in common_splits:
            for stype in score_types:
                visualize_archive_together(
                    domains,
                    exp_dir,
                    trunc_its=args.trunc_its,
                    split=split,
                    type=stype,
                    plot_borders=args.plot_borders,
                    save_svg=args.svg,
                )
    else:
        for domain in domains:
            ensemble_domain = args.check_ensemble and can_domain_ensembled(domain)
            splits = get_domain_splits(domain)
            score_types = ["agent", "ensemble", "max"] if ensemble_domain else ["agent"]
            for split in splits:  # pyright: ignore
                for stype in score_types:
                    visualize_archive_single(
                        domain,
                        exp_dir,
                        trunc_its=args.trunc_its,
                        split=split,
                        type=stype,
                        plot_borders=args.plot_borders,
                        save_svg=args.svg,
                    )
