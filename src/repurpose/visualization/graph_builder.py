import html
from collections import deque

from pyvis.network import Network


def neighborhood(triples, root, hops=2, max_nodes=100):
    selected, frontier, edges = {root}, deque([(root, 0)]), []
    while frontier:
        node, depth = frontier.popleft()
        if depth >= hops:
            continue
        adjacent = triples[(triples["head"] == node) | (triples["tail"] == node)].sort_values(
            ["relation", "head", "tail"]
        )
        for h, r, t in adjacent.itertuples(index=False, name=None):
            other = t if h == node else h
            if other not in selected:
                if len(selected) >= max_nodes:
                    continue
                selected.add(other)
                frontier.append((other, depth + 1))
            edges.append((h, r, t))
    return selected, sorted(set(edges))


def graph_html(entities, triples, root, max_nodes=100):
    nodes, edges = neighborhood(triples, root, max_nodes=max_nodes)
    lookup = entities.set_index("entity_id").to_dict("index")
    network = Network(
        height="460px",
        width="100%",
        bgcolor="#0e1726",
        font_color="#e4edf7",
        directed=True,
        cdn_resources="in_line",
    )
    colors = {"disease": "#f8b84a", "drug": "#40c8b5", "gene/protein": "#8fa3ff", "pathway": "#ee85b5"}
    shapes = {"disease": "diamond", "drug": "dot", "gene/protein": "square", "pathway": "triangle"}
    for node in sorted(nodes):
        info = lookup.get(node, {"name": node, "type": "unknown"})
        network.add_node(
            node,
            label=str(info["name"])[:45],
            title=html.escape(node),
            color=colors.get(info["type"], "#b8c2cf"),
            shape=shapes.get(info["type"], "dot"),
        )
    for h, r, t in edges:
        network.add_edge(h, t, title=html.escape(r), label=r)
    network.set_options('{"physics":{"stabilization":{"iterations":80}},"edges":{"font":{"size":10}}}')
    return network.generate_html()


def local_paths(triples, disease, drug, limit=8):
    first = triples[(triples["head"] == disease) | (triples["tail"] == disease)]
    direct, paths = [], []
    for h, r, t in first.itertuples(index=False, name=None):
        middle = t if h == disease else h
        if middle == drug:
            direct.append(f"{disease} —[{r}]— {drug}")
            continue
        second = triples[
            ((triples["head"] == middle) & (triples["tail"] == drug))
            | ((triples["tail"] == middle) & (triples["head"] == drug))
        ]
        for _, r2, _ in second.itertuples(index=False, name=None):
            paths.append(f"{disease} —[{r}]— {middle} —[{r2}]— {drug}")
    return sorted(set(direct + paths))[:limit]
