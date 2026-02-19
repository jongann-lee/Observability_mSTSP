"""
Automatic map generator for benchmarking RepeatedTopK vs Shortest Path agents.

Design: "Grow blobs from centers."
1. Sample evenly-spread blob centers on the grid
2. Grow blobs outward from centers (distance-ordered → roughly convex shapes)
3. Only remove blob-boundary edges (corridor network stays fully 4-connected)
4. Re-add 2-3 entry edges per blob
5. Place chokepoints on corridor edges adjacent to blobs, ≥30% into path

This matches the handcrafted benchmark philosophy: large organic blobs with
narrow gaps between them, and a fully-connected corridor network.
"""

import sys
import os
import heapq
import numpy as np
import networkx as nx

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Graph_Generation.height_graph_generation import HeightMapGrid


# ---------------------------------------------------------------------------
#  Step 1: Sample blob centers
# ---------------------------------------------------------------------------

def _sample_centers(grid_size, num_blobs, source, target, rng,
                    min_sep_fraction=0.30, margin=2):
    """
    Sample blob centers that are evenly spread across the grid.

    - Centers are at least `min_sep_fraction * grid_size` apart (Manhattan)
    - Centers are at least `margin` away from source and target
    - Uses rejection sampling with up to 500 attempts per center
    """
    min_sep = max(3, int(min_sep_fraction * grid_size))
    forbidden = set()
    for dr in range(-margin, margin + 1):
        for dc in range(-margin, margin + 1):
            for base in [source, target]:
                r, c = base[0] + dr, base[1] + dc
                if 0 <= r < grid_size and 0 <= c < grid_size:
                    forbidden.add((r, c))

    centers = []
    for _ in range(num_blobs):
        placed = False
        for _attempt in range(500):
            r = rng.randint(2, grid_size - 2)
            c = rng.randint(2, grid_size - 2)
            candidate = (r, c)

            if candidate in forbidden:
                continue

            # Check min separation from existing centers
            too_close = False
            for existing in centers:
                if abs(candidate[0] - existing[0]) + abs(candidate[1] - existing[1]) < min_sep:
                    too_close = True
                    break
            if too_close:
                continue

            centers.append(candidate)
            placed = True
            break

        if not placed:
            raise ValueError(
                f"Could not place center {len(centers)+1}/{num_blobs} "
                f"(grid={grid_size}, min_sep={min_sep})")

    return centers


# ---------------------------------------------------------------------------
#  Step 2: Grow blobs from centers
# ---------------------------------------------------------------------------

def _grow_blobs(grid_size, centers, target_coverage, source, target, rng):
    """
    Grow blobs outward from centers using distance-ordered BFS.

    Prioritizing nodes closest to center produces roughly convex shapes.
    A gap of ≥1 node is maintained between blobs.

    Returns list of blobs (each blob is a list of (row, col) tuples).
    """
    total_nodes = grid_size * grid_size
    total_target = int(target_coverage * total_nodes)
    num_blobs = len(centers)

    # Randomize per-blob target sizes (±25% of equal share)
    base_size = total_target // num_blobs
    target_sizes = []
    for _ in range(num_blobs):
        variation = rng.uniform(0.75, 1.25)
        target_sizes.append(max(5, int(base_size * variation)))

    # Track which nodes are claimed or forbidden
    claimed = {}  # node → blob_index
    forbidden = {source, target}  # can never be blob nodes

    # Reserve a 1-node-wide corridor along grid edges so perimeter L-shaped
    # paths (like the handcrafted map) are never blocked by blobs.
    for i in range(grid_size):
        forbidden.add((0, i))            # top row
        forbidden.add((grid_size - 1, i))  # bottom row
        forbidden.add((i, 0))            # left col
        forbidden.add((i, grid_size - 1))  # right col

    # For each blob, maintain a priority queue: (distance_to_center, random_tiebreak, node)
    blob_nodes = [[] for _ in range(num_blobs)]
    queues = []
    for bi, center in enumerate(centers):
        blob_nodes[bi].append(center)
        claimed[center] = bi
        pq = []
        r, c = center
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                dist = abs(nr - center[0]) + abs(nc - center[1])
                heapq.heappush(pq, (dist, rng.rand(), (nr, nc)))
        queues.append(pq)

    # Grow blobs in round-robin, one node at a time
    active = list(range(num_blobs))
    rounds_without_progress = 0

    while active and rounds_without_progress < num_blobs * 3:
        progress = False
        next_active = []

        for bi in active:
            if len(blob_nodes[bi]) >= target_sizes[bi]:
                continue

            center = centers[bi]
            added = False

            while queues[bi]:
                dist, _, node = heapq.heappop(queues[bi])
                r, c = node

                if node in claimed or node in forbidden:
                    continue

                # Gap constraint: node must not be 4-adjacent to a different blob
                adjacent_to_other = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    nb = (nr, nc)
                    if nb in claimed and claimed[nb] != bi:
                        adjacent_to_other = True
                        break
                if adjacent_to_other:
                    continue

                # Claim this node
                claimed[node] = bi
                blob_nodes[bi].append(node)
                added = True
                progress = True

                # Add neighbors to queue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        nb = (nr, nc)
                        if nb not in claimed and nb not in forbidden:
                            d = abs(nr - center[0]) + abs(nc - center[1])
                            heapq.heappush(queues[bi], (d, rng.rand(), nb))
                break  # One node per blob per round

            if len(blob_nodes[bi]) < target_sizes[bi]:
                next_active.append(bi)

        active = next_active
        if not progress:
            rounds_without_progress += 1
        else:
            rounds_without_progress = 0

    # Filter out trivially small blobs
    blobs = [b for b in blob_nodes if len(b) >= 3]
    return blobs


# ---------------------------------------------------------------------------
#  Step 3: Entry point selection
# ---------------------------------------------------------------------------

def _pick_spread_entries(boundary_edges, all_blob_set, num_entries):
    """
    Greedy farthest-point strategy for spatially spread entry points.
    """
    edge_corridor_nodes = []
    for edge in boundary_edges:
        u, v = edge
        corridor_node = v if u in all_blob_set else u
        edge_corridor_nodes.append((edge, corridor_node))

    if len(edge_corridor_nodes) <= num_entries:
        return [e for e, _ in edge_corridor_nodes]

    chosen = [edge_corridor_nodes[0]]
    remaining = list(edge_corridor_nodes[1:])

    while len(chosen) < num_entries and remaining:
        best_idx = -1
        best_min_dist = -1
        for i, (edge, cnode) in enumerate(remaining):
            min_dist = min(
                abs(cnode[0] - ch[0]) + abs(cnode[1] - ch[1])
                for _, ch in chosen
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        if best_idx >= 0:
            chosen.append(remaining.pop(best_idx))
        else:
            break

    return [e for e, _ in chosen]


# ---------------------------------------------------------------------------
#  Step 4: Chokepoint placement
# ---------------------------------------------------------------------------

def _find_chokepoints(graph, source, target, all_blob_nodes, corridor_edges,
                      num_chokepoints=18, rng=None, min_path_fraction=0.30):
    """
    Places chokepoints on corridor edges that are:
    1. Part of the corridor network
    2. Spatially adjacent to a blob
    3. >= min_path_fraction along the shortest path
    4. Grouped into chains of 2-3
    """
    if rng is None:
        rng = np.random.RandomState()

    grid_size = max(max(n) for n in graph.nodes()) + 1

    try:
        sp = nx.shortest_path(graph, source, target, weight='distance')
        sp_cost = sum(graph.edges[sp[i], sp[i + 1]]['distance']
                      for i in range(len(sp) - 1))
    except nx.NetworkXNoPath:
        return []

    dist_from_source = nx.single_source_dijkstra_path_length(
        graph, source, weight='distance')

    # Diverse paths for edge frequency
    diverse_paths = []
    temp_g = graph.copy()
    for _ in range(20):
        try:
            p = nx.shortest_path(temp_g, source, target, weight='distance')
            diverse_paths.append(p)
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                if temp_g.has_edge(u, v):
                    temp_g.edges[u, v]['distance'] *= 3.0
        except nx.NetworkXNoPath:
            break

    edge_freq = {}
    for path in diverse_paths:
        for i in range(len(path) - 1):
            ek = tuple(sorted((path[i], path[i + 1])))
            edge_freq[ek] = edge_freq.get(ek, 0) + 1

    # Spatial blob adjacency
    blob_adjacent = set()
    for r, c in all_blob_nodes:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if (nr, nc) not in all_blob_nodes:
                    blob_adjacent.add((nr, nc))

    min_dist = sp_cost * min_path_fraction

    # Near source/target exclusion
    near_st = set()
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            for base in [source, target]:
                nr, nc = base[0] + dr, base[1] + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    near_st.add((nr, nc))

    # Score candidate edges: must be corridor, blob-adjacent, on a path.
    # Key: compute "detour cost" = how much removing an edge increases SP cost.
    scored = []
    for ek in corridor_edges:
        u, v = ek
        if not graph.has_edge(u, v) and not graph.has_edge(v, u):
            continue
        if u in all_blob_nodes or v in all_blob_nodes:
            continue

        freq = edge_freq.get(ek, 0)
        if freq == 0:
            continue

        # Must be spatially adjacent to blob
        if u not in blob_adjacent and v not in blob_adjacent:
            continue

        edge_dist = min(dist_from_source.get(u, 0), dist_from_source.get(v, 0))
        if edge_dist < min_dist:
            continue
        if u in near_st and v in near_st:
            continue

        # Compute detour cost: remove this edge, recompute SP
        detour_bonus = 1.0
        test_g = graph.copy()
        if test_g.has_edge(u, v):
            test_g.remove_edge(u, v)
        elif test_g.has_edge(v, u):
            test_g.remove_edge(v, u)
        try:
            alt_sp = nx.shortest_path(test_g, source, target, weight='distance')
            alt_cost = sum(test_g.edges[alt_sp[i], alt_sp[i+1]]['distance']
                           for i in range(len(alt_sp) - 1))
            cost_increase = alt_cost - sp_cost
            if cost_increase > 0.5:
                detour_bonus = 1.0 + cost_increase
        except nx.NetworkXNoPath:
            detour_bonus = 5.0  # Bridge edge

        score = float(freq) * detour_bonus
        if u in blob_adjacent and v in blob_adjacent:
            score *= 2.0
        if graph.degree(u) == 2 and graph.degree(v) == 2:
            score *= 2.0
        elif graph.degree(u) == 2 or graph.degree(v) == 2:
            score *= 1.5

        path_frac = edge_dist / sp_cost if sp_cost > 0 else 0
        if 0.4 <= path_frac <= 0.85:
            score *= 1.5

        scored.append((ek, score))

    scored.sort(key=lambda x: -x[1])

    # Identify which diverse path each edge belongs to (for route diversity)
    edge_to_paths = {}
    for pi, path in enumerate(diverse_paths):
        for i in range(len(path) - 1):
            ek = tuple(sorted((path[i], path[i + 1])))
            edge_to_paths.setdefault(ek, set()).add(pi)

    # Build chains of 2-3, ensuring CPs are distributed across routes
    chosen = []
    used = set()
    paths_covered = set()

    # First pass: pick from uncovered routes to ensure diversity
    for ek, sc in scored:
        if len(chosen) >= num_chokepoints:
            break
        if ek in used:
            continue

        # Prefer edges on routes not yet covered
        edge_paths = edge_to_paths.get(ek, set())
        uncovered = edge_paths - paths_covered
        # Skip if all this edge's routes are already covered AND we have
        # fewer than half the CPs (force diversity in first half)
        if not uncovered and len(chosen) < num_chokepoints // 2:
            continue

        u, v = ek
        chain = _extend_chain(graph, u, v, all_blob_nodes, blob_adjacent,
                              corridor_edges, used, target_length=3)
        if not chain:
            continue

        test_g = graph.copy()
        test_g.remove_edges_from(chain)
        if not nx.has_path(test_g, source, target):
            continue

        for e in chain:
            chosen.append(e)
            used.add(tuple(sorted(e)))
            for pi in edge_to_paths.get(tuple(sorted(e)), set()):
                paths_covered.add(pi)

    # Second pass: fill remaining slots without diversity constraint
    if len(chosen) < num_chokepoints:
        for ek, sc in scored:
            if len(chosen) >= num_chokepoints:
                break
            if ek in used:
                continue

            u, v = ek
            chain = _extend_chain(graph, u, v, all_blob_nodes, blob_adjacent,
                                  corridor_edges, used, target_length=3)
            if not chain:
                continue

            test_g = graph.copy()
            test_g.remove_edges_from(chain)
            if not nx.has_path(test_g, source, target):
                continue

            for e in chain:
                chosen.append(e)
                used.add(tuple(sorted(e)))

    return chosen


def _extend_chain(graph, u, v, all_blob_nodes, blob_adjacent,
                  corridor_edges, used_edges, target_length=3):
    """Extend edge into a chain of 2-3 corridor edges adjacent to blobs.

    IMPORTANT: Every edge in the chain must have at least one endpoint
    that is spatially adjacent to a blob, so validation passes.
    """
    chain = [(u, v)]
    used_in_chain = {tuple(sorted((u, v)))}

    def _is_edge_blob_adjacent(n1, n2):
        """Check that at least one endpoint of the edge is blob-adjacent."""
        return n1 in blob_adjacent or n2 in blob_adjacent

    # Forward from v
    current, prev = v, u
    while len(chain) < target_length:
        best_next = None
        best_score = -1
        for n in graph.neighbors(current):
            if n == prev or n in all_blob_nodes:
                continue
            ek = tuple(sorted((current, n)))
            if ek in used_edges or ek in used_in_chain:
                continue
            if ek not in corridor_edges:
                continue
            # Must be blob-adjacent to pass validation
            if not _is_edge_blob_adjacent(current, n):
                continue
            s = 2.0 if n in blob_adjacent else 1.0
            if s > best_score:
                best_score = s
                best_next = n
        if best_next is None:
            break
        chain.append((current, best_next))
        used_in_chain.add(tuple(sorted((current, best_next))))
        prev, current = current, best_next

    if len(chain) >= 2:
        return chain

    # Backward from u
    current, prev = u, v
    while len(chain) < target_length:
        best_next = None
        best_score = -1
        for n in graph.neighbors(current):
            if n == prev or n in all_blob_nodes:
                continue
            ek = tuple(sorted((current, n)))
            if ek in used_edges or ek in used_in_chain:
                continue
            if ek not in corridor_edges:
                continue
            # Must be blob-adjacent to pass validation
            if not _is_edge_blob_adjacent(current, n):
                continue
            s = 2.0 if n in blob_adjacent else 1.0
            if s > best_score:
                best_score = s
                best_next = n
        if best_next is None:
            break
        chain.insert(0, (best_next, current))
        used_in_chain.add(tuple(sorted((best_next, current))))
        prev, current = current, best_next

    if len(chain) >= 2:
        return chain
    return None


# ---------------------------------------------------------------------------
#  Main generation
# ---------------------------------------------------------------------------

def generate_corridor_map(grid_size=12, num_blobs=4, blob_coverage=0.45,
                          num_chokepoints=18, block_prob=0.5, seed=None):
    """
    Generates a map by growing blobs from random centers.

    Only blob-boundary edges are removed. The corridor network (all edges
    between non-blob nodes) stays fully 4-connected.
    """
    rng = np.random.RandomState(seed)
    source = (0, 0)
    target = (grid_size - 1, grid_size - 1)

    # Step 1: Sample blob centers
    centers = _sample_centers(grid_size, num_blobs, source, target, rng)

    # Step 2: Grow blobs from centers
    blobs = _grow_blobs(grid_size, centers, blob_coverage, source, target, rng)

    if len(blobs) < 2:
        raise ValueError("Not enough blobs grown.")

    all_blob_set = set()
    for blob in blobs:
        all_blob_set.update(blob)

    # Step 3: Create HeightMapGrid
    map_gen = HeightMapGrid(m=grid_size, n=grid_size)
    for blob in blobs:
        map_gen.add_plataeu(blob)
    map_gen.calculate_distances()
    map_gen.calculate_simple_visibility(blobs)

    # Only remove blob-boundary edges (one endpoint blob, one corridor).
    # DO NOT remove corridor-corridor edges.
    edges_to_remove = []
    for u, v in list(map_gen.G.edges()):
        u_blob = u in all_blob_set
        v_blob = v in all_blob_set
        # Boundary edge: exactly one endpoint is a blob node
        if u_blob != v_blob:
            edges_to_remove.append((u, v))

    removed_attrs = {}
    for u, v in edges_to_remove:
        if map_gen.G.has_edge(u, v):
            ek = tuple(sorted((u, v)))
            removed_attrs[ek] = dict(map_gen.G.edges[u, v])
            map_gen.G.remove_edge(u, v)

    # Step 4: Re-add 2-3 entry edges per blob
    blob_boundary = {}
    for ek, attrs in removed_attrs.items():
        u, v = ek
        u_blob = u in all_blob_set
        blob_node = u if u_blob else v
        for bi, blob in enumerate(blobs):
            if blob_node in set(blob):
                blob_boundary.setdefault(bi, []).append(ek)
                break

    for bi, edges in blob_boundary.items():
        blob_size = len(blobs[bi])
        num_entries = 2 if blob_size < 12 else 3
        num_entries = min(num_entries, len(edges))

        if num_entries == 0:
            continue

        chosen = _pick_spread_entries(edges, all_blob_set, num_entries)
        for edge in chosen:
            u, v = edge
            if not map_gen.G.has_edge(u, v):
                map_gen.G.add_edge(u, v, **removed_attrs[edge])

    # Verify connectivity
    if not nx.has_path(map_gen.G, source, target):
        raise ValueError("Graph disconnected after construction.")

    # Recompute visibility (filter out removed edges)
    map_gen.calculate_simple_visibility(blobs)
    for node in map_gen.G.nodes():
        if "visible_edges" in map_gen.G.nodes[node]:
            vis = map_gen.G.nodes[node]["visible_edges"]
            map_gen.G.nodes[node]["visible_edges"] = [
                e for e in vis
                if map_gen.G.has_edge(e[0], e[1]) or map_gen.G.has_edge(e[1], e[0])
            ]

    env_graph = map_gen.get_graph()

    # Step 5: Compute corridor_edges = all edges between non-blob nodes
    corridor_edges = set()
    for u, v in env_graph.edges():
        if u not in all_blob_set and v not in all_blob_set:
            corridor_edges.add(tuple(sorted((u, v))))

    # Step 6: Place chokepoints
    chokepoints = _find_chokepoints(
        env_graph, source, target, all_blob_set, corridor_edges,
        num_chokepoints=num_chokepoints, rng=rng,
        min_path_fraction=0.30
    )

    valid_chokepoints = []
    for u, v in chokepoints:
        if env_graph.has_edge(u, v):
            if 'distance' not in env_graph.edges[u, v]:
                h_u = env_graph.nodes[u]['height']
                h_v = env_graph.nodes[v]['height']
                env_graph.edges[u, v]['distance'] = 1.0 + 2.0 * abs(h_u - h_v)
            if 'observed_edge' not in env_graph.edges[u, v]:
                env_graph.edges[u, v]['observed_edge'] = False
            valid_chokepoints.append((u, v))

    return {
        'env_graph': env_graph,
        'blobs': blobs,
        'chokepoints': valid_chokepoints,
        'source': source,
        'target': target,
        'block_prob': block_prob,
        'seed': seed,
        'grid_size': grid_size,
    }


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------

def _validate_map(map_data, min_chokepoints=6, min_cp_coverage=0.5,
                  num_blockage_trials=30, min_sp_cost_cv=0.03):
    g = map_data['env_graph']
    s, t = map_data['source'], map_data['target']
    chokepoints = map_data['chokepoints']
    block_prob = map_data.get('block_prob', 0.5)

    if not nx.has_path(g, s, t):
        return False
    if len(chokepoints) < min_chokepoints:
        return False

    all_blob_nodes = set()
    for blob in map_data['blobs']:
        all_blob_nodes.update(blob)

    # Every CP must be spatially adjacent to a blob
    for u, v in chokepoints:
        u_adj = any((u[0] + dr, u[1] + dc) in all_blob_nodes
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
        v_adj = any((v[0] + dr, v[1] + dc) in all_blob_nodes
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
        if not u_adj and not v_adj:
            return False

    # SP should mostly avoid blobs
    sp = nx.shortest_path(g, s, t, weight="distance")
    blob_frac = sum(1 for n in sp if n in all_blob_nodes) / len(sp)
    if blob_frac > 0.3:
        return False

    # Chokepoint coverage
    cp_set = set(tuple(sorted(e)) for e in chokepoints)
    temp_g = g.copy()
    paths_through_cp = 0
    total_paths = 0
    for _ in range(10):
        try:
            p = nx.shortest_path(temp_g, s, t, weight="distance")
            total_paths += 1
            path_edges = set(tuple(sorted((p[i], p[i + 1])))
                             for i in range(len(p) - 1))
            if path_edges & cp_set:
                paths_through_cp += 1
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                if temp_g.has_edge(u, v):
                    temp_g.edges[u, v]['distance'] *= 3.0
        except nx.NetworkXNoPath:
            break
    if total_paths == 0:
        return False
    if (paths_through_cp / total_paths) < min_cp_coverage:
        return False

    # Blockage impact
    rng_val = np.random.RandomState(map_data.get('seed', 0) + 9999)
    sp_costs = []
    baseline_sp = nx.shortest_path(g, s, t, weight="distance")
    baseline_cost = sum(g.edges[baseline_sp[i], baseline_sp[i + 1]]['distance']
                        for i in range(len(baseline_sp) - 1))
    total_detours = 0
    for _ in range(num_blockage_trials):
        edges_to_remove = []
        rolls = rng_val.rand(len(chokepoints))
        for i, edge in enumerate(chokepoints):
            if rolls[i] < block_prob:
                u, v = edge
                if g.has_edge(u, v):
                    edges_to_remove.append((u, v))
                elif g.has_edge(v, u):
                    edges_to_remove.append((v, u))
        if not edges_to_remove:
            sp_costs.append(baseline_cost)
            continue
        blocked_g = g.copy()
        blocked_g.remove_edges_from(edges_to_remove)
        if not nx.has_path(blocked_g, s, t):
            continue
        try:
            blocked_sp = nx.shortest_path(blocked_g, s, t, weight="distance")
            cost = sum(blocked_g.edges[blocked_sp[i], blocked_sp[i + 1]]['distance']
                       for i in range(len(blocked_sp) - 1))
            sp_costs.append(cost)
            if cost > baseline_cost + 0.5:
                total_detours += 1
        except (nx.NetworkXNoPath, KeyError):
            continue

    if len(sp_costs) < num_blockage_trials // 2:
        return False
    sp_arr = np.array(sp_costs)
    if np.mean(sp_arr) <= 0:
        return False
    cv = np.std(sp_arr) / np.mean(sp_arr)
    if cv < min_sp_cost_cv:
        return False
    if total_detours < 2:
        return False

    return True


# ---------------------------------------------------------------------------
#  Suite generation
# ---------------------------------------------------------------------------

def generate_map_suite(num_maps=50, seed_start=1000):
    maps = []
    configs = [
        # (grid_size, num_blobs, blob_coverage, num_chokepoints, block_prob, label)
        (12, 4, 0.45, 18, 0.50, "standard_12x12"),
        (12, 4, 0.45, 21, 0.50, "dense_cp_12x12"),
        (14, 4, 0.43, 18, 0.50, "standard_14x14"),
        (14, 4, 0.43, 21, 0.50, "dense_cp_14x14"),
        (16, 5, 0.40, 21, 0.50, "standard_16x16"),
        (16, 5, 0.40, 24, 0.50, "dense_cp_16x16"),
        (12, 4, 0.45, 18, 0.40, "lowblock_12x12"),
        (14, 4, 0.43, 18, 0.40, "lowblock_14x14"),
        (16, 5, 0.40, 21, 0.40, "lowblock_16x16"),
        (14, 6, 0.43, 18, 0.50, "six_blobs_14x14"),
    ]
    maps_per_config = max(1, num_maps // len(configs))
    remainder = num_maps - maps_per_config * len(configs)

    map_id = 0
    for cfg_idx, (gs, nb, bc, nc, bp, label) in enumerate(configs):
        count = maps_per_config + (1 if cfg_idx < remainder else 0)
        generated = 0
        attempts = 0
        max_attempts = count * 100
        while generated < count and attempts < max_attempts:
            seed = seed_start + map_id
            attempts += 1
            try:
                map_data = generate_corridor_map(
                    grid_size=gs, num_blobs=nb, blob_coverage=bc,
                    num_chokepoints=nc, block_prob=bp, seed=seed
                )
                map_data['label'] = label
                map_data['map_id'] = map_id
                if _validate_map(map_data):
                    maps.append(map_data)
                    generated += 1
                map_id += 1
            except Exception:
                map_id += 1
                continue
    return maps


if __name__ == "__main__":
    map_data = generate_corridor_map(grid_size=12, num_blobs=4, seed=42)
    g = map_data['env_graph']

    all_blob_nodes = set()
    for blob in map_data['blobs']:
        all_blob_nodes.update(blob)

    gs = map_data['grid_size']
    print(f"Grid: {gs}x{gs}")
    print(f"Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
    print(f"Blob nodes: {len(all_blob_nodes)} "
          f"({len(all_blob_nodes) / g.number_of_nodes() * 100:.1f}%)")
    print(f"Blobs: {len(map_data['blobs'])}, "
          f"sizes: {sorted([len(b) for b in map_data['blobs']], reverse=True)}")
    print(f"Chokepoints: {len(map_data['chokepoints'])}")

    # Entry edges per blob
    for bi, blob in enumerate(map_data['blobs']):
        bset = set(blob)
        entry_count = 0
        for u, v in g.edges():
            u_in = u in bset
            v_in = v in bset
            if u_in != v_in and (u_in or v_in):
                entry_count += 1
        print(f"  Blob {bi} (size={len(blob)}): {entry_count} entries")

    # Total entry edges
    entry = sum(1 for u, v in g.edges()
                if (u in all_blob_nodes) != (v in all_blob_nodes))
    print(f"Total entry edges: {entry}")

    # Spatial blob adjacency
    blob_adj = set()
    for r, c in all_blob_nodes:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < gs and 0 <= nc < gs and (nr, nc) not in all_blob_nodes:
                blob_adj.add((nr, nc))
    non_blob = g.number_of_nodes() - len(all_blob_nodes)
    print(f"Corridor nodes spatially adj to blob: {len(blob_adj)}/{non_blob} "
          f"({len(blob_adj) / non_blob * 100:.0f}%)")

    # CP adjacency
    cp_adj = sum(1 for u, v in map_data['chokepoints']
                 if any((u[0]+dr, u[1]+dc) in all_blob_nodes
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
                 or any((v[0]+dr, v[1]+dc) in all_blob_nodes
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]))
    print(f"Chokepoints adj to blob: {cp_adj}/{len(map_data['chokepoints'])}")

    sp = nx.shortest_path(g, map_data['source'], map_data['target'],
                          weight="distance")
    sp_len = sum(g.edges[sp[i], sp[i+1]]['distance'] for i in range(len(sp)-1))
    print(f"SP: length={sp_len:.0f}, hops={len(sp)-1}")

    dist_from_source = nx.single_source_dijkstra_path_length(
        g, map_data['source'], weight='distance')
    for u, v in map_data['chokepoints']:
        d = min(dist_from_source.get(u, 0), dist_from_source.get(v, 0))
        frac = d / sp_len if sp_len > 0 else 0
        print(f"  CP ({u},{v}): dist={d:.1f}, frac={frac:.2f}")

    print(f"\nValid: {_validate_map(map_data)}")
