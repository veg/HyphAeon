"""
WHY THIS FILE EXISTS

Reference values for js/src/numeric/graph.js and the numpy pieces of js/src/sectors.js, generated
by the libraries hyphaeon/epistasis.py actually calls — networkx 3.6.1 under CPython 3.14 and
numpy — so the JavaScript ports are measured against the exact ordering and tie-breaking of the
reference, not against a reading of its source. fixtures/ (scripts/gen_fixtures.py) covers
extract_epistatic_sectors_tse end to end on ONE planted graph whose two communities never tie; this
file is where ties, set-order iteration and the merge order are exercised on hundreds of small random
graphs. It is NOT a fixture in the fixtures/README.md sense: it lives with the tests, and
js/test/numeric-graph.test.js / js/test/sectors.test.js read its output.

Regenerate from js/ with the reference environment (the venv that has hyphaeon installed):

    python test/data/sectors/gen.py

Output: graph.json in this directory.

What is recorded:

  set_order        list(set(values)) for random lists of small non-negative ints — CPython's set
                   iteration order, which networkx's subgraph views iterate in when the induced node
                   set is smaller than half the graph (FilterAtlas.__iter__)
  subgraph_order   list(G.subgraph(nodes)) and list(G.subgraph(nodes)[n]) for every n, on graphs
                   with nodes 1..L and random induced sets, including a hub whose neighbour list is
                   longer than twice the induced set (the neighbour-level set-order branch)
  components       [list(c) for c in nx.connected_components(G)] on random sparse graphs with
                   shuffled node insertion, plus the components of small induced subgraphs
  communities      [sorted(c) for c in greedy_modularity_communities(G, weight=...)] in the returned
                   order, on random weighted graphs, integer-weighted graphs (ties), unweighted
                   graphs, complete graphs and cycles (every pair ties), two cliques with a bridge,
                   disconnected graphs (the StopIteration path), the cutoff / best_n / resolution
                   arguments, and induced subgraphs of a 1,097-node graph (the set-order path)
  percentile       np.percentile / np.mean / np.std of float32 arrays, the null reductions of
                   compute_sector_permutation_test
  sectors          extract_epistatic_sectors_tse (hyphaeon itself, n_permutations=0) on variations the
                   fixtures do not cover: min_clique_size above the node count (the components path
                   with two components), a min_coherence that drops one sector, and a 1,097-node graph
                   with two identical triangles whose sector ids are decided by the set order
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities

HERE = Path(__file__).resolve().parent
SEED = 20260905


def jsonable(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    return v


def gen_set_order(rng: random.Random):
    cases = []
    for size in list(range(1, 25)) + [30, 37, 40, 50, 64, 70, 100, 150]:
        for upper in (8, 40, 100, 1097, 5000):
            if size > upper:
                continue
            vals = rng.sample(range(upper), size)
            cases.append({"values": vals, "order": list(set(vals))})
    # duplicates collapse
    for _ in range(20):
        vals = [rng.randrange(0, 60) for _ in range(rng.randrange(1, 40))]
        cases.append({"values": vals, "order": list(set(vals))})
    # resize boundaries: consecutive sizes around fill*5 >= mask*3
    for size in range(1, 80):
        vals = rng.sample(range(2000), size)
        cases.append({"values": vals, "order": list(set(vals))})
    return cases


class RecordingGraph(nx.Graph):
    """nx.Graph that remembers the ORDER edges were added in. G.edges() iterates node order then
    adjacency, which is not the insertion order, and the JavaScript side rebuilds the graph by
    replaying add_node / add_edge — so what is written must be the calls, not the view."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.added = []

    def add_edge(self, u, v, **attr):
        self.added.append([u, v, dict(attr)])
        super().add_edge(u, v, **attr)


def graph_to_json(G):
    edges = G.added if isinstance(G, RecordingGraph) else [[u, v, dict(d)] for u, v, d in G.edges(data=True)]
    return {
        "nodes": [[n, dict(d)] for n, d in G.nodes(data=True)],
        "edges": edges,
    }


def random_graph(rng: random.Random, n: int, p: float, labels=None, weights="float", shuffle=True):
    labels = list(labels) if labels is not None else list(range(1, n + 1))
    if shuffle:
        rng.shuffle(labels)
    G = RecordingGraph()
    G.add_nodes_from(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                if weights == "float":
                    G.add_edge(labels[i], labels[j], weight=round(rng.uniform(0.05, 1.0), 4))
                elif weights == "int":
                    G.add_edge(labels[i], labels[j], weight=rng.randrange(1, 4))
                elif weights == "none":
                    G.add_edge(labels[i], labels[j])
                elif weights == "mixed":
                    if rng.random() < 0.5:
                        G.add_edge(labels[i], labels[j], weight=round(rng.uniform(0.05, 1.0), 4))
                    else:
                        G.add_edge(labels[i], labels[j])
    return G


def gen_subgraph_order(rng: random.Random):
    """Graphs with nodes 1..L in order; the edge lists (insertion order) are shared under `graphs`."""
    graphs = {}
    cases = []
    for L in (12, 40, 300, 1097):
        G = RecordingGraph()
        G.add_nodes_from(range(1, L + 1))
        # hub 1 connected to everything, plus a sprinkling of random edges
        for v in range(2, L + 1):
            G.add_edge(1, v, weight=1.0)
        for _ in range(L):
            u, v = rng.sample(range(1, L + 1), 2)
            G.add_edge(u, v, weight=round(rng.uniform(0.1, 1.0), 3))
        graphs[str(L)] = G.added
        for k in (2, 3, 5, 6, 8, 13, 21, L // 2, L // 2 + 1, L):
            if k > L:
                continue
            sub = rng.sample(range(1, L + 1), k)
            if rng.random() < 0.7 and 1 not in sub:
                sub[0] = 1
            H = G.subgraph(sub)
            cases.append({
                "L": L,
                "nodes": sub,
                "node_order": list(H),
                "neighbor_order": {str(n): list(H[n]) for n in H},
            })
    return {"graphs": graphs, "cases": cases}


def gen_components(rng: random.Random):
    cases = []
    for _ in range(60):
        n = rng.randrange(2, 30)
        G = random_graph(rng, n, rng.choice([0.05, 0.1, 0.15, 0.3]), weights="none")
        cases.append({"graph": graph_to_json(G), "components": [list(c) for c in nx.connected_components(G)]})
    # induced subgraphs of a big graph: the set-order branch decides component order
    L = 1097
    G = RecordingGraph()
    G.add_nodes_from(range(1, L + 1))
    for _ in range(600):
        u, v = rng.sample(range(1, L + 1), 2)
        G.add_edge(u, v)
    subs = []
    for _ in range(25):
        k = rng.randrange(2, 40)
        sub = rng.sample(range(1, L + 1), k)
        # make sure there are some edges inside
        for _ in range(k // 2):
            u, v = rng.sample(sub, 2)
            G.add_edge(u, v)
        subs.append(sub)
    for sub in subs:
        H = G.subgraph(sub)
        cases.append({"big": True, "nodes": sub, "L": L,
                      "components": [list(c) for c in nx.connected_components(H)]})
    return {"big_edges": G.added, "cases": cases}


def communities_case(G, name, **kw):
    weight = kw.get("weight", None)
    comms = greedy_modularity_communities(G, **kw)
    return {"name": name, "graph": graph_to_json(G), "kwargs": {k: v for k, v in kw.items()},
            "communities": [sorted(c) for c in comms]}


def gen_communities(rng: random.Random):
    cases = []
    # random float-weighted graphs
    for i in range(80):
        n = rng.randrange(3, 28)
        G = random_graph(rng, n, rng.choice([0.15, 0.25, 0.4, 0.6]), weights="float")
        if G.number_of_edges() == 0:
            continue
        cases.append(communities_case(G, f"float_{i}", weight="weight"))
    # integer weights: many ties
    for i in range(60):
        n = rng.randrange(3, 24)
        G = random_graph(rng, n, rng.choice([0.2, 0.35, 0.6]), weights="int")
        if G.number_of_edges() == 0:
            continue
        cases.append(communities_case(G, f"int_{i}", weight="weight"))
    # unweighted graphs, weight=None and weight='weight' with the attribute missing (both read 1)
    for i in range(40):
        n = rng.randrange(3, 24)
        G = random_graph(rng, n, rng.choice([0.2, 0.35, 0.6]), weights="none")
        if G.number_of_edges() == 0:
            continue
        cases.append(communities_case(G, f"unweighted_none_{i}"))
        cases.append(communities_case(G, f"unweighted_attr_{i}", weight="weight"))
    # mixed: some edges carry a weight, some do not
    for i in range(20):
        n = rng.randrange(4, 20)
        G = random_graph(rng, n, 0.4, weights="mixed")
        if G.number_of_edges() == 0:
            continue
        cases.append(communities_case(G, f"mixed_{i}", weight="weight"))
    # symmetric graphs: every candidate merge ties
    for n in range(3, 13):
        cases.append(communities_case(nx.complete_graph(range(1, n + 1)), f"complete_{n}", weight="weight"))
        cases.append(communities_case(nx.cycle_graph(range(1, n + 1)), f"cycle_{n}", weight="weight"))
        cases.append(communities_case(nx.path_graph(range(1, n + 1)), f"path_{n}"))
        cases.append(communities_case(nx.star_graph(n), f"star_{n}"))
    for n in range(2, 8):
        cases.append(communities_case(nx.complete_bipartite_graph(n, n), f"bipartite_{n}"))
    # two cliques joined by one bridge, equal weights
    for a in range(3, 7):
        G = RecordingGraph()
        for i in range(a):
            for j in range(i + 1, a):
                G.add_edge(i + 1, j + 1, weight=1.0)
                G.add_edge(a + i + 1, a + j + 1, weight=1.0)
        G.add_edge(a, a + 1, weight=1.0)
        cases.append(communities_case(G, f"two_cliques_{a}", weight="weight"))
        G2 = RecordingGraph()
        for u, v, d in G.added:
            G2.add_edge(u, v, **d)
        G2.add_edge(1, 2 * a, weight=1.0)
        cases.append(communities_case(G2, f"two_cliques_two_bridges_{a}", weight="weight"))
    # disconnected graphs (StopIteration path), with equal-size components
    for i in range(30):
        G = RecordingGraph()
        base = 1
        for _ in range(rng.randrange(2, 5)):
            size = rng.randrange(2, 6)
            nodes = list(range(base, base + size))
            base += size
            for x in range(len(nodes)):
                for y in range(x + 1, len(nodes)):
                    if rng.random() < 0.8:
                        G.add_edge(nodes[x], nodes[y], weight=round(rng.uniform(0.2, 1.0), 3))
            for u in nodes:
                G.add_node(u)
        if G.number_of_edges() == 0:
            continue
        cases.append(communities_case(G, f"disconnected_{i}", weight="weight"))
    # cutoff / best_n / resolution arguments
    for i in range(15):
        n = rng.randrange(6, 20)
        G = random_graph(rng, n, 0.35, weights="float")
        if G.number_of_edges() == 0:
            continue
        cases.append(communities_case(G, f"cutoff_{i}", weight="weight", cutoff=rng.randrange(1, 4)))
        cases.append(communities_case(G, f"best_n_{i}", weight="weight", best_n=rng.randrange(1, 4)))
        cases.append(communities_case(G, f"resolution_{i}", weight="weight", resolution=rng.choice([0.5, 1.5])))
    # the epistasis shape: nodes 1..1097 inserted in order, a few active nodes, subgraph view
    L = 1097
    for i in range(20):
        Gi = RecordingGraph()
        Gi.add_nodes_from(range(1, L + 1))
        k = rng.randrange(3, 30)
        active = rng.sample(range(1, L + 1), k)
        for x in range(k):
            for y in range(x + 1, k):
                if rng.random() < 0.4:
                    Gi.add_edge(active[x], active[y], weight=round(rng.uniform(0.3, 1.0), 4))
        sub_nodes = [n for n, d in Gi.degree() if d > 0]
        if len(sub_nodes) < 2:
            continue
        H = Gi.subgraph(sub_nodes)
        comms = greedy_modularity_communities(H, weight="weight")
        cases.append({"name": f"subgraph_1097_{i}", "graph": None, "L": L,
                      "edges": Gi.added,
                      "sub_nodes": sub_nodes, "kwargs": {"weight": "weight"},
                      "communities": [sorted(c) for c in comms]})
    # same shape but heavily tied integer weights
    for i in range(10):
        Gi = RecordingGraph()
        Gi.add_nodes_from(range(1, L + 1))
        k = rng.randrange(4, 24)
        active = rng.sample(range(1, L + 1), k)
        for x in range(k):
            for y in range(x + 1, k):
                if rng.random() < 0.5:
                    Gi.add_edge(active[x], active[y], weight=1.0)
        sub_nodes = [n for n, d in Gi.degree() if d > 0]
        if len(sub_nodes) < 2:
            continue
        H = Gi.subgraph(sub_nodes)
        comms = greedy_modularity_communities(H, weight="weight")
        cases.append({"name": f"subgraph_1097_ties_{i}", "graph": None, "L": L,
                      "edges": Gi.added,
                      "sub_nodes": sub_nodes, "kwargs": {"weight": "weight"},
                      "communities": [sorted(c) for c in comms]})
    return cases


def gen_percentile(nrng: np.random.Generator):
    cases = []
    for n in (1, 2, 3, 5, 19, 20, 21, 100, 2000, 5000):
        a = nrng.random(n).astype(np.float32)
        cases.append({"values": a, "p95": np.percentile(a, 95), "mean": np.mean(a), "std": np.std(a)})
    return cases


def gen_sectors():
    from hyphaeon.epistasis import extract_epistatic_sectors_tse

    root = HERE.parents[3]
    fixture = json.loads((root / "fixtures" / "epistasis" / "extract_epistatic_sectors_tse.json").read_text())
    base = next(c for c in fixture if c["name"] == "planted_no_permutations")["inputs"]

    def build(gj):
        G = nx.Graph()
        for n, a in gj["nodes"]:
            G.add_node(n, **a)
        for u, v, a in gj["edges"]:
            G.add_edge(u, v, **a)
        return G

    attr = np.asarray(base["attributions"], dtype=np.float32)
    lrts = np.asarray(base["lrts"], dtype=np.float32)
    aas = base["consensus_aas"]
    cases = []
    G = build(base["graph"])
    for name, kw in [("min_clique_size_100_components_path", dict(min_clique_size=100)),
                     ("min_coherence_drops_one", dict(min_coherence=0.9993)),
                     ("min_clique_size_100_min_coherence", dict(min_clique_size=100, min_coherence=0.9993))]:
        secs = extract_epistatic_sectors_tse(G, attr, lrts, aas, n_permutations=0, **kw)
        cases.append({"name": name, "graph": base["graph"], "attributions": attr, "lrts": lrts, "consensus_aas": aas,
                      "kwargs": kw, "sectors": secs})
    # two identical triangles on a 1,097-node graph: equal size, equal coherence; the set order of
    # {5, 6, 7, 33, 34, 35} (table of 32 slots: 33 -> slot 1, 34 -> 2, 35 -> 3) decides the ids
    L = 1097
    N = 6
    A = np.zeros((L, N), dtype=np.float32)
    pattern = np.array([[0.9, 0.8, 0.0, 0.7, 0.0, 0.1], [0.85, 0.75, 0.05, 0.6, 0.0, 0.0], [0.8, 0.9, 0.0, 0.65, 0.0, 0.2]], np.float32)
    for tri in ([5, 6, 7], [33, 34, 35]):
        for k, site in enumerate(tri):
            A[site - 1] = pattern[k]
    lr = np.linspace(0.5, 9.0, L).astype(np.float32)
    aa = [chr(ord("A") + (i % 20)) for i in range(L)]
    G = nx.Graph()
    G.add_nodes_from(range(1, L + 1))
    for tri in ([5, 6, 7], [33, 34, 35]):
        G.add_edge(tri[0], tri[1], weight=1.0)
        G.add_edge(tri[0], tri[2], weight=1.0)
        G.add_edge(tri[1], tri[2], weight=1.0)
    secs = extract_epistatic_sectors_tse(G, A, lr, aa, n_permutations=0)
    cases.append({"name": "two_identical_triangles_1097_set_order", "graph": graph_to_json(G), "attributions": A,
                  "lrts": lr, "consensus_aas": aa, "kwargs": {}, "sectors": secs})
    secs = extract_epistatic_sectors_tse(G, A, lr, aa, n_permutations=0, min_clique_size=100)
    cases.append({"name": "two_identical_triangles_1097_components_order", "graph": graph_to_json(G), "attributions": A,
                  "lrts": lr, "consensus_aas": aa, "kwargs": {"min_clique_size": 100}, "sectors": secs})
    return cases


def main():
    rng = random.Random(SEED)
    nrng = np.random.default_rng(SEED)
    payload = {
        "generator": "js/test/data/sectors/gen.py",
        "python": sys.version.split()[0],
        "networkx": nx.__version__,
        "numpy": np.__version__,
        "set_order": gen_set_order(rng),
        "subgraph_order": gen_subgraph_order(rng),
        "components": gen_components(rng),
        "communities": gen_communities(rng),
        "percentile": gen_percentile(nrng),
        "sectors": gen_sectors(),
    }
    path = HERE / "graph.json"
    path.write_text(json.dumps(jsonable(payload), separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size} bytes); communities cases: {len(payload['communities'])}")


if __name__ == "__main__":
    main()
