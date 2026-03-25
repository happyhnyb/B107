import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity


def canon(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", name.lower().strip())


def load_data(data_dir: Path):
    with open(data_dir / "knowledge_graph_entities.json", "r", encoding="utf-8") as f:
        entities = json.load(f)
    with open(data_dir / "knowledge_graph_relationships.json", "r", encoding="utf-8") as f:
        relationships = json.load(f)
    entities_df = pd.DataFrame(entities)
    rels_df = pd.DataFrame(relationships)
    entities_df["metadata_dict"] = entities_df["metadata"].apply(
        lambda x: json.loads(x) if isinstance(x, str) and x else {}
    )
    entities_df["mention_count"] = entities_df["metadata_dict"].apply(lambda d: d.get("ds10_mention_count"))
    entities_df["person_type"] = entities_df["metadata_dict"].apply(lambda d: d.get("person_type"))
    entities_df["occupation"] = entities_df["metadata_dict"].apply(lambda d: d.get("occupation"))
    entities_df["legal_status"] = entities_df["metadata_dict"].apply(lambda d: d.get("legal_status"))
    entities_df["canon_name"] = entities_df["name"].map(canon)
    return entities_df, rels_df


def build_clean_graph(entities_df: pd.DataFrame, rels_df: pd.DataFrame):
    id_to_canon = entities_df.set_index("id")["canon_name"].to_dict()
    canon_rows = []
    for canon_name, grp in entities_df.groupby("canon_name"):
        names = grp["name"].dropna().tolist()
        chosen = sorted(names, key=lambda s: (-len(str(s)), str(s)))[0] if names else canon_name
        entity_type = grp["entity_type"].mode().iloc[0]
        person_type = grp["person_type"].mode().iloc[0] if grp["person_type"].notna().any() else None
        mention_max = pd.to_numeric(grp["mention_count"], errors="coerce").max()
        canon_rows.append(
            {
                "canon_name": canon_name,
                "display_name": chosen,
                "entity_type": entity_type,
                "person_type": person_type,
                "mention_count_max": mention_max,
                "merged_source_nodes": len(grp),
            }
        )
    canon_df = pd.DataFrame(canon_rows)

    graph = nx.Graph()
    for _, row in canon_df.iterrows():
        graph.add_node(
            row["canon_name"],
            name=row["display_name"],
            entity_type=row["entity_type"],
            person_type=row["person_type"],
            mention_count=row["mention_count_max"],
            merged_source_nodes=row["merged_source_nodes"],
        )

    for _, row in rels_df.iterrows():
        s = id_to_canon.get(row["source_entity_id"])
        t = id_to_canon.get(row["target_entity_id"])
        if not s or not t or s == t:
            continue
        weight = float(row["weight"]) if row["weight"] is not None else 1.0
        if graph.has_edge(s, t):
            graph[s][t]["weight"] += weight
            graph[s][t]["relationship_types"].add(row["relationship_type"])
            graph[s][t]["edge_count"] += 1
        else:
            graph.add_edge(
                s,
                t,
                weight=weight,
                relationship_types={row["relationship_type"]},
                primary_relationship=row["relationship_type"],
                edge_count=1,
            )
    return graph, canon_df


def compute_outputs(graph: nx.Graph, canon_df: pd.DataFrame, rels_df: pd.DataFrame):
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest = graph.subgraph(components[0]).copy()

    degree = dict(largest.degree())
    weighted_degree = dict(largest.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(largest, weight="weight", normalized=True)
    closeness = nx.closeness_centrality(largest)
    eigenvector = nx.eigenvector_centrality_numpy(largest, weight="weight")
    pagerank = nx.pagerank(largest, weight="weight")
    clustering = nx.clustering(largest)

    communities = louvain_communities(largest, weight="weight", seed=42)
    comm_map = {}
    for i, community in enumerate(sorted(communities, key=len, reverse=True), start=1):
        for node in community:
            comm_map[node] = i

    centrality_df = pd.DataFrame(
        {
            "canon_name": list(largest.nodes()),
            "name": [largest.nodes[n]["name"] for n in largest.nodes()],
            "entity_type": [largest.nodes[n]["entity_type"] for n in largest.nodes()],
            "person_type": [largest.nodes[n].get("person_type") for n in largest.nodes()],
            "mention_count": [largest.nodes[n].get("mention_count") for n in largest.nodes()],
            "merged_source_nodes": [largest.nodes[n].get("merged_source_nodes") for n in largest.nodes()],
            "degree": [degree[n] for n in largest.nodes()],
            "weighted_degree": [weighted_degree[n] for n in largest.nodes()],
            "betweenness": [betweenness[n] for n in largest.nodes()],
            "closeness": [closeness[n] for n in largest.nodes()],
            "eigenvector": [eigenvector[n] for n in largest.nodes()],
            "pagerank": [pagerank[n] for n in largest.nodes()],
            "clustering": [clustering[n] for n in largest.nodes()],
            "community_id": [comm_map[n] for n in largest.nodes()],
        }
    ).sort_values(["weighted_degree", "betweenness"], ascending=False)

    community_rows = []
    for cid, nodes in sorted(((i, c) for i, c in enumerate(sorted(communities, key=len, reverse=True), start=1)), key=lambda x: len(x[1]), reverse=True):
        subgraph = largest.subgraph(nodes)
        top_connector = (
            centrality_df[centrality_df["community_id"] == cid]
            .sort_values("weighted_degree", ascending=False)
            .iloc[0]["name"]
        )
        community_rows.append(
            {
                "community_id": cid,
                "nodes": len(nodes),
                "edges": subgraph.number_of_edges(),
                "density": nx.density(subgraph) if len(nodes) > 1 else 0,
                "top_connector": top_connector,
            }
        )
    community_df = pd.DataFrame(community_rows)

    rel_type_counts = rels_df["relationship_type"].value_counts().rename_axis("relationship_type").reset_index(name="count")
    entity_type_counts = canon_df["entity_type"].value_counts().rename_axis("entity_type").reset_index(name="count")

    summary_df = pd.DataFrame(
        [
            {
                "raw_nodes": int(len(entities_df)),
                "raw_relationship_rows": int(len(rels_df)),
                "cleaned_nodes": int(graph.number_of_nodes()),
                "cleaned_edges": int(graph.number_of_edges()),
                "connected_components": int(len(components)),
                "largest_component_nodes": int(largest.number_of_nodes()),
                "largest_component_edges": int(largest.number_of_edges()),
                "largest_component_density": float(nx.density(largest)),
                "largest_component_avg_degree": float(sum(dict(largest.degree()).values()) / largest.number_of_nodes()),
                "largest_component_avg_clustering": float(nx.average_clustering(largest)),
                "largest_component_transitivity": float(nx.transitivity(largest)),
                "largest_component_diameter": int(nx.diameter(largest)),
                "largest_component_avg_path_length": float(nx.average_shortest_path_length(largest)),
                "louvain_communities": int(len(communities)),
                "louvain_modularity": float(modularity(largest, communities, weight="weight")),
            }
        ]
    )
    return largest, centrality_df, community_df, rel_type_counts, entity_type_counts, summary_df


def synthetic_comparison(largest: nx.Graph) -> pd.DataFrame:
    n = largest.number_of_nodes()
    m = largest.number_of_edges()
    p = 2 * m / (n * (n - 1))
    k = round((2 * m) / n)
    if k % 2 == 1:
        k += 1

    graphs = {
        "Empirical": largest,
        "ER": nx.gnm_random_graph(n, m, seed=42),
        "BA": nx.barabasi_albert_graph(n, max(1, round(m / n)), seed=42),
        "WS": nx.watts_strogatz_graph(n, max(2, k), min(0.1, max(0.01, p * 5)), seed=42),
    }

    rows = []
    for graph_name, graph in graphs.items():
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        largest_component = graph.subgraph(components[0]).copy()
        rows.append(
            {
                "graph": graph_name,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "components": len(components),
                "largest_component_nodes": largest_component.number_of_nodes(),
                "density": nx.density(graph),
                "avg_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
                "avg_clustering": nx.average_clustering(largest_component),
                "diameter_lcc": nx.diameter(largest_component),
                "avg_path_length_lcc": nx.average_shortest_path_length(largest_component),
            }
        )
    return pd.DataFrame(rows)


def make_figures(largest: nx.Graph, centrality_df: pd.DataFrame, community_df: pd.DataFrame, rel_type_counts: pd.DataFrame, synthetic_df: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(rel_type_counts["relationship_type"], rel_type_counts["count"])
    plt.xticks(rotation=35, ha="right")
    plt.title("Relationship type distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "relationship_type_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(list(dict(largest.degree()).values()), bins=25)
    plt.title("Degree distribution (largest connected component)")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(fig_dir / "degree_distribution.png", dpi=200)
    plt.close()

    top10 = centrality_df.sort_values("weighted_degree", ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    plt.barh(list(reversed(top10["name"].tolist())), list(reversed(top10["weighted_degree"].tolist())))
    plt.title("Top 10 nodes by weighted degree")
    plt.tight_layout()
    plt.savefig(fig_dir / "top10_weighted_degree.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(community_df["community_id"].astype(str), community_df["nodes"])
    plt.title("Louvain community sizes")
    plt.xlabel("Community ID")
    plt.ylabel("Nodes")
    plt.tight_layout()
    plt.savefig(fig_dir / "community_sizes.png", dpi=200)
    plt.close()

    top_nodes = set(centrality_df.sort_values("weighted_degree", ascending=False).head(120)["canon_name"])
    subgraph = largest.subgraph(top_nodes).copy()
    pos = nx.spring_layout(subgraph, seed=42, k=0.55)
    sizes = [40 + 4 * subgraph.degree(n, weight="weight") for n in subgraph.nodes()]
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(subgraph, pos, alpha=0.15, width=0.5)
    nx.draw_networkx_nodes(subgraph, pos, node_size=sizes, alpha=0.85)
    threshold = np.percentile([subgraph.degree(x, weight="weight") for x in subgraph.nodes()], 85)
    labels = {n: subgraph.nodes[n]["name"] for n in subgraph.nodes() if subgraph.degree(n, weight="weight") >= threshold}
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=7)
    plt.title("Largest component subgraph (top 120 nodes by weighted degree)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_dir / "network_subgraph_top120.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    x = np.arange(len(synthetic_df))
    width = 0.35
    plt.bar(x - width / 2, synthetic_df["avg_clustering"], width, label="Avg clustering")
    plt.bar(x + width / 2, synthetic_df["avg_path_length_lcc"], width, label="Avg path length (LCC)")
    plt.xticks(x, synthetic_df["graph"])
    plt.title("Empirical vs synthetic graph comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "synthetic_comparison.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze the curated Epstein knowledge graph.")
    parser.add_argument("--data_dir", default="data", help="Directory with knowledge_graph_entities.json and knowledge_graph_relationships.json")
    parser.add_argument("--output_dir", default="outputs", help="Directory for CSV and PNG outputs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    global entities_df
    entities_df, rels_df = load_data(data_dir)
    graph, canon_df = build_clean_graph(entities_df, rels_df)
    largest, centrality_df, community_df, rel_type_counts, entity_type_counts, summary_df = compute_outputs(graph, canon_df, rels_df)
    synthetic_df = synthetic_comparison(largest)
    make_figures(largest, centrality_df, community_df, rel_type_counts, synthetic_df, fig_dir)

    canon_df.to_csv(output_dir / "cleaned_nodes.csv", index=False)
    centrality_df.to_csv(output_dir / "centrality_rankings.csv", index=False)
    centrality_df[["name", "entity_type", "person_type", "weighted_degree", "degree", "betweenness", "pagerank", "community_id"]].head(25).to_csv(output_dir / "top_nodes_by_weighted_degree.csv", index=False)
    community_df.to_csv(output_dir / "community_summary.csv", index=False)
    rel_type_counts.to_csv(output_dir / "relationship_type_counts.csv", index=False)
    entity_type_counts.to_csv(output_dir / "entity_type_counts.csv", index=False)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    synthetic_df.to_csv(output_dir / "synthetic_graph_comparison.csv", index=False)

    print("Analysis complete.")
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
