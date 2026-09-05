/**
 * WHY THIS FILE EXISTS
 *
 * The two networkx algorithms `hyphaeon/epistasis.py` calls inside `extract_epistatic_sectors_tse`
 * (epistasis.py:327-337 at cf838ab), ported with networkx's ORDERING and TIE-BREAKING, which
 * PLAN.md §5.1 makes part of the specification ("greedy modularity communities (Clauset–Newman–Moore,
 * networkx semantics)"):
 *
 *   connectedComponents           networkx 3.6.1 algorithms/components/connected.py
 *                                 `connected_components` + `_plain_bfs` (lines 13-69, 267-282)
 *   greedyModularityCommunities   networkx 3.6.1 algorithms/community/modularity_max.py
 *                                 `greedy_modularity_communities` (lines 231-386) and its generator
 *                                 `_greedy_modularity_communities_generator` (lines 18-228), on top of
 *                                 a port of `networkx/utils/mapped_queue.py` (`MappedQueue`,
 *                                 `_HeapElement`)
 *   inducedSubgraph               `Graph.subgraph(nodes)` as an ORDERED, materialised adjacency:
 *                                 classes/coreviews.py `FilterAtlas.__iter__` decides whether the view
 *                                 iterates in the parent graph's order or in the order of the Python
 *                                 `set` behind `filters.show_nodes`
 *   cpythonIntSetOrder            the iteration order of a CPython 3.14 `set` of non-negative ints
 *                                 built by adding the values in sequence (Objects/setobject.c:
 *                                 `set_add_entry`, `set_table_resize`, `set_insert_clean`)
 *   adjacencyFromGraph            `nx.Graph.add_node` / `add_edge` insertion semantics over the
 *                                 `{nodes: [[id, attrs]], edges: [[u, v, attrs]]}` shape
 *                                 scripts/gen_fixtures.py `graph_to_json` writes
 *
 * WHY THE ORDER MATTERS. `extract_epistatic_sectors_tse` numbers sectors in the order the
 * communities come back and its final sort, `(coherence, size)` descending, is Python's stable sort,
 * so two communities of equal size and equal coherence keep the order networkx returned them in.
 * networkx returns communities sorted by size (stable) from a dict whose key order is the node
 * iteration order of the graph, and iterates a subgraph view either in the parent's node order or
 * — when the induced node set holds fewer than half the parent's nodes — in the order of a Python
 * `set` of the node ids. For a 1,097-site gene whose co-selection graph has a handful of active
 * nodes that is the set order, and a set of small ints iterates by `id mod table_size` with linear
 * probing, not by value. This file reproduces that so sector ids and tie order match the reference
 * for ANY node labelling; for labels that are not non-negative integers (never the case in
 * epistasis.py, whose nodes are 1..L) CPython's order is hash-randomised per process and this file
 * uses insertion order, which is as reproducible as anything.
 *
 * THE MERGE ORDER. Clauset–Newman–Moore picks the pair with the largest modularity gain dq at each
 * step; networkx breaks ties on the (u, v) node pair, smallest first (`_HeapElement.__lt__`
 * compares priorities, then elements). Floating-point dq values are formed in exactly networkx's
 * operation order (`deg * q0 * 0.5`, `q0 * wt - resolution * (a[u] * b[v] + b[u] * a[v])`, and the
 * three update forms), because which pairs TIE is decided by those roundings. The heap layout
 * itself is irrelevant to the result — every element is a distinct (u, v) pair under a total order —
 * but the MappedQueue is ported as written so the code reads against networkx line for line.
 *
 * WHAT IT DELIBERATELY DOES NOT DO. Directed graphs and multigraphs (the co-selection graph is a
 * simple undirected Graph); `naive_greedy_modularity_communities`; node attributes (an adjacency
 * holds edge attribute objects only, shared between both directions like networkx's edge data
 * dicts). Communities are returned as arrays sorted by node — networkx returns frozensets whose
 * iteration order is not part of any contract, and epistasis.py sorts the members anyway.
 */

/** @typedef {number|string} NodeId */
/**
 * An undirected graph as networkx stores it: node -> (neighbour -> edge attribute object), both
 * levels in insertion order, one attribute object per edge shared by its two directions.
 * @typedef {Map<NodeId, Map<NodeId, Record<string, any>>>} Adjacency
 */

// ---- CPython set iteration order ----------------------------------------------------------------

const PYSET_MINSIZE = 8;
const LINEAR_PROBES = 9;
const PERTURB_SHIFT = 5;

/** Objects/setobject.c `set_insert_clean` (no dummies in the table). */
function pySetInsertClean(table, mask, hash) {
	let perturb = hash;
	let i = hash % (mask + 1);
	for (;;) {
		if (table[i] === -1) {
			table[i] = hash;
			return;
		}
		if (i + LINEAR_PROBES <= mask) {
			for (let j = 1; j <= LINEAR_PROBES; j++) {
				if (table[i + j] === -1) {
					table[i + j] = hash;
					return;
				}
			}
		}
		perturb = Math.floor(perturb / 2 ** PERTURB_SHIFT);
		i = (i * 5 + 1 + perturb) % (mask + 1);
	}
}

/**
 * The iteration order of a CPython 3.14 `set` built by `set(iterable)` from non-negative integers
 * (each value added with `set_add_entry`, tables resized to the smallest power of two above
 * `4 * used` once `fill * 5 >= mask * 3`, entries reinserted in old-table order; `hash(n) == n` for
 * 0 <= n < 2^61 - 1). Duplicates collapse as in a set. Values must be safe non-negative integers
 * below 2^31 for the simulation to be exact; anything else makes the function return the distinct
 * values in insertion order (CPython's order for strings is randomised per process, so nothing
 * could be reproduced there).
 *
 * @param {Iterable<NodeId>} values
 * @returns {NodeId[]}
 */
export function cpythonIntSetOrder(values) {
	const list = Array.from(values);
	const distinct = Array.from(new Set(list));
	for (const v of distinct) {
		if (typeof v !== 'number' || !Number.isInteger(v) || v < 0 || v >= 2 ** 31) return distinct;
	}
	let mask = PYSET_MINSIZE - 1;
	/** @type {number[]} slot -> hash, -1 for empty */
	let table = new Array(PYSET_MINSIZE).fill(-1);
	let used = 0;
	for (const key of /** @type {number[]} */ (list)) {
		// set_add_entry
		const hash = key;
		let perturb = hash;
		let i = hash % (mask + 1);
		let slot = -1;
		let present = false;
		for (;;) {
			let probes = i + LINEAR_PROBES <= mask ? LINEAR_PROBES : 0;
			let e = i;
			do {
				if (table[e] === -1) {
					slot = e;
					break;
				}
				if (table[e] === hash) {
					present = true;
					break;
				}
				e++;
			} while (probes-- > 0);
			if (slot >= 0 || present) break;
			perturb = Math.floor(perturb / 2 ** PERTURB_SHIFT);
			i = (i * 5 + 1 + perturb) % (mask + 1);
		}
		if (present) continue;
		table[slot] = hash;
		used++;
		const fill = used; // no dummies: fill == used
		if (fill * 5 < mask * 3) continue;
		// set_table_resize
		const minused = used > 50000 ? used * 2 : used * 4;
		let newsize = PYSET_MINSIZE;
		while (newsize <= minused) newsize <<= 1;
		const old = table;
		table = new Array(newsize).fill(-1);
		mask = newsize - 1;
		for (let s = 0; s < old.length; s++) if (old[s] !== -1) pySetInsertClean(table, mask, old[s]);
	}
	/** @type {number[]} */
	const out = [];
	for (let s = 0; s < table.length; s++) if (table[s] !== -1) out.push(table[s]);
	return out;
}

/** True when every label is a number (the case for epistasis.py's 1..L nodes). */
function allNumeric(nodes) {
	for (const n of nodes) if (typeof n !== 'number') return false;
	return true;
}

/** Ascending for numeric labels, insertion order otherwise. */
function sortedMembers(nodes) {
	const arr = Array.from(nodes);
	if (allNumeric(arr)) arr.sort((a, b) => /** @type {number} */ (a) - /** @type {number} */ (b));
	return arr;
}

// ---- graph construction ---------------------------------------------------------------------------

/**
 * @typedef {{nodes?: Array<NodeId | [NodeId, Record<string, any>?]>, edges: Array<[NodeId, NodeId, Record<string, any>?]>}} GraphJson
 */

/**
 * `nx.Graph()` fed `add_node(n, **attrs)` for each node then `add_edge(u, v, **attrs)` for each edge
 * (scripts/gen_fixtures.py `graph_from_json`): an endpoint not yet present is appended to the node
 * order at that moment; re-adding an edge updates its attributes in place. A Map passed in is
 * returned as is, and so is the `adj` Map of any object that carries one (epistasis.js's
 * `CoselectionGraph` stores `G._adj` under that name), so a graph built in this library needs no
 * round trip through JSON.
 *
 * @param {GraphJson | Adjacency | {adj: Adjacency}} graph
 * @returns {Adjacency}
 */
export function adjacencyFromGraph(graph) {
	if (graph instanceof Map) return graph;
	if (graph && /** @type {{adj?: unknown}} */ (graph).adj instanceof Map) return /** @type {{adj: Adjacency}} */ (graph).adj;
	/** @type {Adjacency} */
	const adj = new Map();
	for (const entry of graph.nodes ?? []) {
		const id = Array.isArray(entry) ? entry[0] : entry;
		if (!adj.has(id)) adj.set(id, new Map());
	}
	for (const [u, v, attrs] of graph.edges ?? []) {
		if (!adj.has(u)) adj.set(u, new Map());
		if (!adj.has(v)) adj.set(v, new Map());
		const un = /** @type {Map<NodeId, Record<string, any>>} */ (adj.get(u));
		const existing = un.get(v);
		if (existing) {
			Object.assign(existing, attrs ?? {});
		} else {
			const datadict = { ...(attrs ?? {}) };
			un.set(v, datadict);
			/** @type {Map<NodeId, Record<string, any>>} */ (adj.get(v)).set(u, datadict);
		}
	}
	return adj;
}

/**
 * `G.number_of_edges()` — `sum(len(nbrs) for nbrs in adj.values()) // 2` (a self loop counts once
 * in its own row, so the integer division floors it away when it is the only odd contribution).
 *
 * @param {Adjacency} adj
 * @returns {number}
 */
export function numberOfEdges(adj) {
	let s = 0;
	for (const nbrs of adj.values()) s += nbrs.size;
	return Math.floor(s / 2);
}

/**
 * `G.degree(n, weight=weight)`: the number of neighbours (a self loop counted twice) or, with a
 * weight attribute, the sequential sum of `d.get(weight, 1)` over the neighbours in adjacency order
 * plus the self-loop weight once more.
 *
 * @param {Adjacency} adj
 * @param {NodeId} n
 * @param {string | null} [weight]
 * @returns {number}
 */
export function degree(adj, n, weight = null) {
	const nbrs = adj.get(n);
	if (!nbrs) throw new RangeError(`degree: node ${String(n)} is not in the graph`);
	if (weight === null || weight === undefined) return nbrs.size + (nbrs.has(n) ? 1 : 0);
	let s = 0;
	for (const dd of nbrs.values()) s += edgeWeight(dd, weight);
	const self = nbrs.get(n);
	if (self) s += edgeWeight(self, weight);
	return s;
}

/** `dd.get(weight, 1)`. */
function edgeWeight(dd, weight) {
	const w = dd[weight];
	return w === undefined ? 1 : w;
}

/**
 * `G.size(weight)`: the edge count, or half the sequential sum of weighted degrees in node order.
 * @param {Adjacency} adj
 * @param {string | null} [weight]
 * @returns {number}
 */
export function graphSize(adj, weight = null) {
	if (weight === null || weight === undefined) return numberOfEdges(adj);
	let s = 0;
	for (const n of adj.keys()) s += degree(adj, n, weight);
	return s / 2;
}

/**
 * `G.subgraph(nodes)` materialised in the order the networkx view iterates. At the NODE level
 * `FilterAtlas.__iter__` iterates the `show_nodes` set when `2 * len(nodes) < len(parent)` and the
 * parent's node order filtered by membership otherwise. At the NEIGHBOUR level the order is always
 * the parent's adjacency order: `FilterAdjacency.__getitem__` wraps the node filter in a closure
 * (`new_node_ok`) that has no `.nodes` attribute, so the set branch is never taken there. Nodes
 * absent from the parent are skipped (`nbunch_iter`), duplicates collapse. Edge attribute objects
 * are shared with the parent.
 *
 * @param {Adjacency} adj
 * @param {Iterable<NodeId>} nodes
 * @returns {Adjacency}
 */
export function inducedSubgraph(adj, nodes) {
	/** @type {NodeId[]} */
	const wanted = [];
	for (const n of nodes) if (adj.has(n)) wanted.push(n);
	const nodeSet = new Set(wanted);
	const nodeOrder =
		2 * nodeSet.size < adj.size
			? cpythonIntSetOrder(wanted).filter((n) => adj.has(n))
			: Array.from(adj.keys()).filter((n) => nodeSet.has(n));
	/** @type {Adjacency} */
	const sub = new Map();
	for (const n of nodeOrder) {
		const nbrs = /** @type {Map<NodeId, Record<string, any>>} */ (adj.get(n));
		/** @type {Map<NodeId, Record<string, any>>} */
		const row = new Map();
		for (const [w, dd] of nbrs) if (nodeSet.has(w)) row.set(w, dd);
		sub.set(n, row);
	}
	return sub;
}

// ---- connected components -------------------------------------------------------------------------

/**
 * `nx.connected_components(G)` in order: one component per unseen node in node iteration order,
 * grown by `_plain_bfs` (level by level, neighbours in adjacency order, early exit once every node
 * is seen). Each component is returned in the iteration order of the Python `seen` set (insertion
 * order for non-integer labels).
 *
 * @param {Adjacency} adj
 * @returns {NodeId[][]}
 */
export function connectedComponents(adj) {
	const seen = new Set();
	/** @type {NodeId[][]} */
	const out = [];
	const total = adj.size;
	for (const v of adj.keys()) {
		if (seen.has(v)) continue;
		const c = plainBfs(adj, total - seen.size, v);
		for (const w of c) seen.add(w);
		out.push(cpythonIntSetOrder(c));
	}
	return out;
}

/** connected.py:267-282 `_plain_bfs`; returns the `seen` set in insertion order. */
function plainBfs(adj, n, source) {
	const seen = new Set([source]);
	let nextlevel = [source];
	while (nextlevel.length > 0) {
		const thislevel = nextlevel;
		nextlevel = [];
		for (const v of thislevel) {
			for (const w of /** @type {Map<NodeId, any>} */ (adj.get(v)).keys()) {
				if (!seen.has(w)) {
					seen.add(w);
					nextlevel.push(w);
				}
			}
			if (seen.size === n) return seen;
		}
	}
	return seen;
}

// ---- MappedQueue (networkx/utils/mapped_queue.py) -------------------------------------------------

/**
 * `_HeapElement`: priority for ordering, (u, v) for identity. `__lt__` compares priorities and falls
 * back to the element tuple on a tie; `__eq__` / `__hash__` look at the element only.
 * @typedef {{priority: number, u: NodeId, v: NodeId, key: string}} HeapElement
 */

/** @returns {string} the identity of a (u, v) element, `hash((u, v))`'s role */
function pairKey(u, v) {
	return `${typeof u}:${String(u)} ${typeof v}:${String(v)}`;
}

/** @returns {HeapElement} */
function heapElement(priority, u, v) {
	return { priority, u, v, key: pairKey(u, v) };
}

/** `_HeapElement.__lt__`: priority, then the (u, v) tuple. */
function lt(a, b) {
	if (a.priority === b.priority) {
		if (a.u === b.u) return a.v < b.v;
		return a.u < b.u;
	}
	return a.priority < b.priority;
}

/** heapq._siftup (CPython Lib/heapq.py): push heap[pos] down to a leaf, then _siftdown back to pos. */
function heapqSiftup(heap, pos) {
	const endPos = heap.length;
	const startPos = pos;
	const newitem = heap[pos];
	let childPos = 2 * pos + 1;
	while (childPos < endPos) {
		const rightPos = childPos + 1;
		if (rightPos < endPos && !lt(heap[childPos], heap[rightPos])) childPos = rightPos;
		heap[pos] = heap[childPos];
		pos = childPos;
		childPos = 2 * pos + 1;
	}
	heap[pos] = newitem;
	// heapq._siftdown(heap, startpos, pos)
	while (pos > startPos) {
		const parentPos = (pos - 1) >> 1;
		const parent = heap[parentPos];
		if (!lt(newitem, parent)) break;
		heap[pos] = parent;
		pos = parentPos;
	}
	heap[pos] = newitem;
}

/** A min-heap with O(log n) removal and priority update, keyed by element identity. */
class MappedQueue {
	/** @param {HeapElement[]} [data] */
	constructor(data = []) {
		/** @type {HeapElement[]} */
		this.heap = data.slice();
		/** @type {Map<string, number>} */
		this.position = new Map();
		this._heapify();
	}

	_heapify() {
		// mapped_queue.py calls the STDLIB heapq.heapify here, whose _siftup bubbles back up only as
		// far as the position it started from — not the class's own _siftup (which bubbles to the
		// root, and would break the bottom-up construction).
		const h = this.heap;
		for (let i = Math.floor(h.length / 2) - 1; i >= 0; i--) heapqSiftup(h, i);
		this.position = new Map();
		for (let pos = 0; pos < h.length; pos++) this.position.set(h[pos].key, pos);
		if (this.position.size !== h.length) throw new Error('Heap contains duplicate elements');
	}

	get length() {
		return this.heap.length;
	}

	/** @param {HeapElement} elt @returns {boolean} */
	push(elt) {
		if (this.position.has(elt.key)) return false;
		const pos = this.heap.length;
		this.heap.push(elt);
		this.position.set(elt.key, pos);
		this._siftdown(0, pos);
		return true;
	}

	/** @returns {HeapElement} */
	pop() {
		const heap = this.heap;
		const elt = heap[0];
		this.position.delete(elt.key);
		if (heap.length === 1) {
			heap.pop();
			return elt;
		}
		const last = /** @type {HeapElement} */ (heap.pop());
		heap[0] = last;
		this.position.set(last.key, 0);
		this._siftup(0);
		return elt;
	}

	/** Replace element `key` with `next` (possibly the same identity at a new priority). */
	update(key, next) {
		const pos = this.position.get(key);
		if (pos === undefined) throw new Error(`MappedQueue.update: ${key} not in queue`);
		this.heap[pos] = next;
		this.position.delete(key);
		this.position.set(next.key, pos);
		this._siftup(pos);
	}

	/** @param {string} key */
	remove(key) {
		const pos = this.position.get(key);
		if (pos === undefined) throw new Error(`MappedQueue.remove: ${key} not in queue`);
		this.position.delete(key);
		if (pos === this.heap.length - 1) {
			this.heap.pop();
			return;
		}
		const last = /** @type {HeapElement} */ (this.heap.pop());
		this.heap[pos] = last;
		this.position.set(last.key, pos);
		this._siftup(pos);
	}

	/** heapq._siftup with position bookkeeping: move the smaller child up until a leaf, then bubble. */
	_siftup(pos) {
		const heap = this.heap;
		const position = this.position;
		const endPos = heap.length;
		const newitem = heap[pos];
		let childPos = 2 * pos + 1;
		while (childPos < endPos) {
			let child = heap[childPos];
			const rightPos = childPos + 1;
			if (rightPos < endPos) {
				const right = heap[rightPos];
				if (!lt(child, right)) {
					child = right;
					childPos = rightPos;
				}
			}
			heap[pos] = child;
			position.set(child.key, pos);
			pos = childPos;
			childPos = 2 * pos + 1;
		}
		// mapped_queue.py bubbles to the ROOT here (`while pos > 0`), unlike heapq._siftup's
		// `startpos`; update() and remove() at an interior position depend on it.
		while (pos > 0) {
			const parentPos = (pos - 1) >> 1;
			const parent = heap[parentPos];
			if (!lt(newitem, parent)) break;
			heap[pos] = parent;
			position.set(parent.key, pos);
			pos = parentPos;
		}
		heap[pos] = newitem;
		position.set(newitem.key, pos);
	}

	/** heapq._siftdown with position bookkeeping. */
	_siftdown(startPos, pos) {
		const heap = this.heap;
		const position = this.position;
		const newitem = heap[pos];
		while (pos > startPos) {
			const parentPos = (pos - 1) >> 1;
			const parent = heap[parentPos];
			if (!lt(newitem, parent)) break;
			heap[pos] = parent;
			position.set(parent.key, pos);
			pos = parentPos;
		}
		heap[pos] = newitem;
		position.set(newitem.key, pos);
	}
}

// ---- Clauset–Newman–Moore --------------------------------------------------------------------------

/**
 * modularity_max.py:18-228 `_greedy_modularity_communities_generator` for an undirected graph.
 * Yields the live communities Map first, then alternately the dq of the next merge and the Map
 * after it. The Map is the Python dict: `communities[v]` is reassigned in place (keeping v's
 * position) and `communities[u]` deleted, so `values()` order is what networkx sorts.
 *
 * @param {Adjacency} adj
 * @param {string | null} weight
 * @param {number} resolution
 * @returns {Generator<Map<NodeId, Set<NodeId>> | number, void, void>}
 */
function* greedyModularityGenerator(adj, weight, resolution) {
	const m = graphSize(adj, weight);
	const q0 = 1 / m;

	/** @type {Map<NodeId, number>} a == b for undirected graphs */
	const a = new Map();
	for (const node of adj.keys()) a.set(node, degree(adj, node, weight) * q0 * 0.5);
	const A = (n) => /** @type {number} */ (a.get(n));

	// dq_dict: defaultdict(lambda: defaultdict(float)) filled from G.edges(data=weight, default=1)
	/** @type {Map<NodeId, Map<NodeId, number>>} */
	const dqDict = new Map();
	for (const n of adj.keys()) dqDict.set(n, new Map());
	const row = (n) => /** @type {Map<NodeId, number>} */ (dqDict.get(n));
	{
		const seen = new Set();
		for (const [n, nbrs] of adj) {
			for (const [nbr, dd] of nbrs) {
				if (seen.has(nbr)) continue;
				if (n === nbr) continue;
				const wt = weight === null ? 1 : edgeWeight(dd, weight);
				row(n).set(nbr, (row(n).get(nbr) ?? 0) + wt);
				row(nbr).set(n, (row(nbr).get(n) ?? 0) + wt);
			}
			seen.add(n);
		}
	}
	for (const [u, nbrdict] of dqDict) {
		for (const [v, wt] of nbrdict) nbrdict.set(v, q0 * wt - resolution * (A(u) * A(v) + A(u) * A(v)));
	}

	/** @type {Map<NodeId, MappedQueue>} */
	const dqHeap = new Map();
	for (const u of adj.keys()) {
		const elts = [];
		for (const [v, dq] of row(u)) elts.push(heapElement(-dq, u, v));
		dqHeap.set(u, new MappedQueue(elts));
	}
	const H_init = [];
	for (const n of adj.keys()) {
		const hq = /** @type {MappedQueue} */ (dqHeap.get(n));
		if (hq.length > 0) H_init.push(hq.heap[0]);
	}
	const H = new MappedQueue(H_init);
	const heapOf = (n) => /** @type {MappedQueue} */ (dqHeap.get(n));

	/** @type {Map<NodeId, Set<NodeId>>} */
	const communities = new Map();
	for (const n of adj.keys()) communities.set(n, new Set([n]));
	yield communities;

	while (H.length > 1) {
		const top = H.pop();
		const { u, v } = top;
		const dq = -top.priority;
		yield dq;
		heapOf(u).pop();
		if (heapOf(u).length > 0) H.push(heapOf(u).heap[0]);
		const vu = pairKey(v, u);
		if (heapOf(v).heap[0].key === vu) {
			H.remove(vu);
			heapOf(v).remove(vu);
			if (heapOf(v).length > 0) H.push(heapOf(v).heap[0]);
		} else {
			heapOf(v).remove(vu);
		}

		// Perform merge
		const merged = new Set(/** @type {Set<NodeId>} */ (communities.get(u)));
		for (const x of /** @type {Set<NodeId>} */ (communities.get(v))) merged.add(x);
		communities.set(v, merged);
		communities.delete(u);

		const uNbrs = new Set(row(u).keys());
		const vNbrs = new Set(row(v).keys());
		const allNbrs = new Set([...uNbrs, ...vNbrs]);
		allNbrs.delete(u);
		allNbrs.delete(v);
		for (const w of allNbrs) {
			let dqVw;
			if (uNbrs.has(w) && vNbrs.has(w)) {
				dqVw = /** @type {number} */ (row(v).get(w)) + /** @type {number} */ (row(u).get(w));
			} else if (vNbrs.has(w)) {
				dqVw = /** @type {number} */ (row(v).get(w)) - resolution * (A(u) * A(w) + A(w) * A(u));
			} else {
				dqVw = /** @type {number} */ (row(u).get(w)) - resolution * (A(v) * A(w) + A(w) * A(v));
			}
			for (const [r, c] of [
				[v, w],
				[w, v]
			]) {
				const dqHeapRow = heapOf(r);
				row(r).set(c, dqVw);
				const dOldmax = dqHeapRow.length > 0 ? dqHeapRow.heap[0] : null;
				const d = heapElement(-dqVw, r, c);
				if (vNbrs.has(w)) dqHeapRow.update(d.key, d);
				else dqHeapRow.push(d);
				if (dOldmax === null) {
					H.push(d);
				} else {
					const rowMax = dqHeapRow.heap[0];
					if (dOldmax.key !== rowMax.key || dOldmax.priority !== rowMax.priority) H.update(dOldmax.key, rowMax);
				}
			}
		}

		// Remove row/col u from the dq matrix
		for (const w of row(u).keys()) {
			row(w).delete(u);
			if (w !== v) {
				for (const [r, c] of [
					[w, u],
					[u, w]
				]) {
					const dqHeapRow = heapOf(r);
					const dOld = pairKey(r, c);
					if (dqHeapRow.heap[0].key === dOld) {
						dqHeapRow.remove(dOld);
						H.remove(dOld);
						if (dqHeapRow.length > 0) H.push(dqHeapRow.heap[0]);
					} else {
						dqHeapRow.remove(dOld);
					}
				}
			}
		}
		dqDict.delete(u);
		dqHeap.set(u, new MappedQueue());
		a.set(v, A(v) + A(u));
		a.set(u, 0);

		yield communities;
	}
}

/**
 * `nx.community.greedy_modularity_communities(G, weight=None, resolution=1, cutoff=1, best_n=None)`
 * (modularity_max.py:231-386): merge until the next best merge would lower modularity (or `cutoff`
 * communities remain; or, with `bestN`, keep merging the two largest until at most `bestN` remain),
 * then return the communities sorted by size, largest first, ties in the dict order described above.
 * Each community is an array of its members sorted ascending (numeric labels) or in insertion order.
 *
 * @param {Adjacency} adj undirected graph
 * @param {{weight?: string | null, resolution?: number, cutoff?: number, bestN?: number | null}} [options]
 *   `weight` is the edge attribute name (default null: every edge weighs 1; a missing attribute
 *   also reads as 1)
 * @returns {NodeId[][]}
 */
export function greedyModularityCommunities(adj, options = {}) {
	const weight = options.weight === undefined ? null : options.weight;
	const resolution = options.resolution ?? 1;
	const cutoff = options.cutoff ?? 1;
	const N = adj.size;

	if (!graphSize(adj, null)) return Array.from(adj.keys(), (n) => [n]);

	if (cutoff < 1 || cutoff > N) throw new RangeError(`cutoff must be between 1 and ${N}. Got ${cutoff}.`);
	let bestN = options.bestN ?? null;
	if (bestN !== null) {
		if (bestN < 1 || bestN > N) throw new RangeError(`best_n must be between 1 and ${N}. Got ${bestN}.`);
		if (bestN < cutoff) throw new RangeError(`Must have best_n >= cutoff. Got ${bestN} < ${cutoff}`);
		if (bestN === 1) return [sortedMembers(adj.keys())];
	} else {
		bestN = N;
	}

	const gen = greedyModularityGenerator(adj, weight, resolution);
	let communities = /** @type {Map<NodeId, Set<NodeId>>} */ (gen.next().value);

	/** Python's `sorted(communities, key=len, reverse=True)`: stable, dict order among equal sizes. */
	const bySizeDesc = (comms) => {
		const arr = Array.from(comms).map((c, i) => ({ c, i }));
		arr.sort((x, y) => y.c.size - x.c.size || x.i - y.i);
		return arr.map((x) => x.c);
	};

	while (communities.size > cutoff) {
		const step = gen.next();
		if (step.done) {
			// StopIteration: the communities are the connected components.
			let sorted = bySizeDesc(communities.values());
			while (sorted.length > bestN) {
				const [c1, c2, ...rest] = sorted;
				const merged = new Set(c1);
				for (const x of c2) merged.add(x);
				sorted = [merged, ...rest];
			}
			return sorted.map((c) => sortedMembers(c));
		}
		const dq = /** @type {number} */ (step.value);
		if (dq < 0 && communities.size <= bestN) break;
		communities = /** @type {Map<NodeId, Set<NodeId>>} */ (gen.next().value);
	}
	return bySizeDesc(communities.values()).map((c) => sortedMembers(c));
}
