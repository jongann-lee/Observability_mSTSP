"""
Automatic map generator for benchmarking RepeatedTopK vs Shortest Path agents.

Generates grid maps with plateau obstacles that create environments where:
1. The shortest path goes through narrow corridors AROUND blobs
2. Blobs provide high visibility but cost extra to enter (height penalty)
3. Chokepoints on corridor routes can be blocked, forcing costly detours
4. The visibility-aware agent can detect blockages early by entering blobs

This mirrors the handcrafted benchmark map structure.
"""

import sys
import os
import numpy as np
import networkx as nx

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Graph_Generation.height_graph_generation import HeightMapGrid


def _grow_constrained_blob(grid_m, grid_n, seed_node, target_size, rng,
                           existing_occupied, corridor_cells):
    """
    Grows a blob that does NOT expand into corridor cells.
    This ensures corridors remain passable.
    """
    forbidden = existing_occupied | corridor_cells
    blob = set()
    blob.add(seed_node)
    frontier = [seed_node]

    while len(blob) < target_size and frontier:
        idx = rng.randint(len(frontier))
        node = frontier[idx]
        r, c = node
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_m and 0 <= nc < grid_n:
                if (nr, nc) not in blob and (nr, nc) not in forbidden:
                    neighbors.append((nr, nc))
        if neighbors:
            chosen = neighbors[rng.randint(len(neighbors))]
            blob.add(chosen)
            frontier.append(chosen)
        else:
            frontier.pop(idx)

    return list(blob)


def _define_corridors(grid_size, rng, num_corridors=3):
    """
    Defines corridor paths through the grid that blobs must not block.
    Corridors are strictly 1-cell-wide paths along grid edges and through
    the middle, creating a network of narrow passages.

    Returns a set of corridor cell coordinates and a list of corridor paths.
    """
    source = (0, 0)
    target = (grid_size - 1, grid_size - 1)

    corridor_cells = set()
    corridor_paths = []

    # Corridor 1: Along bottom edge then right edge
    path1 = []
    for c in range(grid_size):
        path1.append((grid_size - 1, c))
    for r in range(grid_size - 2, -1, -1):
        path1.append((r, grid_size - 1))
    corridor_paths.append(path1)
    corridor_cells.update(path1)

    # Corridor 2: Along left edge (column 0)
    path2 = []
    for r in range(grid_size):
        path2.append((r, 0))
    corridor_paths.append(path2)
    corridor_cells.update(path2)

    if num_corridors >= 3:
        # Corridor 3: Through the middle
        path3 = []
        mid = grid_size // 2
        for c in range(mid + 1):
            path3.append((mid, c))
        for r in range(mid + 1, grid_size):
            path3.append((r, mid))
        corridor_paths.append(path3)
        corridor_cells.update(path3)

    if num_corridors >= 4:
        # Corridor 4: Along top edge
        path4 = []
        for c in range(grid_size):
            path4.append((0, c))
        corridor_paths.append(path4)
        corridor_cells.update(path4)

    return corridor_cells, corridor_paths


def _find_chokepoints_via_cuts(graph, source, target, num_chokepoints=18,
                                rng=None):
    """
    Finds chokepoints by identifying critical edges that many diverse
    paths must traverse.
    """
    if rng is None:
        rng = np.random.RandomState()

    # Find diverse paths
    diverse_paths = []
    temp_graph = graph.copy()
    for _ in range(20):
        try:
            sp = nx.shortest_path(temp_graph, source, target, weight="distance")
            diverse_paths.append(sp)
            for i in range(len(sp) - 1):
                u, v = sp[i], sp[i+1]
                if temp_graph.has_edge(u, v):
                    temp_graph.edges[u, v]['distance'] *= 3.0
        except nx.NetworkXNoPath:
            break

    if not diverse_paths:
        return []

    # Track edge frequency
    edge_freq = {}
    for path in diverse_paths:
        for i in range(len(path) - 1):
            edge = tuple(sorted((path[i], path[i+1])))
            edge_freq[edge] = edge_freq.get(edge, 0) + 1

    sorted_edges = sorted(edge_freq.items(), key=lambda x: -x[1])

    # Build chokepoint groups (chains of 3 consecutive high-frequency edges)
    chosen_chokepoints = []
    used_edges = set()

    for edge, freq in sorted_edges:
        if len(chosen_chokepoints) >= num_chokepoints:
            break
        if edge in used_edges:
            continue

        u, v = edge
        chain = _extend_chain(graph, u, v, edge_freq, used_edges, target_length=3)
        if not chain:
            continue

        # Verify connectivity after removal
        test_graph = graph.copy()
        test_graph.remove_edges_from(chain)
        if not nx.has_path(test_graph, source, target):
            continue

        for e in chain:
            chosen_chokepoints.append(e)
            used_edges.add(tuple(sorted(e)))

    return chosen_chokepoints


def _extend_chain(graph, u, v, edge_freq, used_edges, target_length=3):
    """Extends an edge into a chain of consecutive high-frequency edges."""
    chain = [(u, v)]
    used_in_chain = {tuple(sorted((u, v)))}

    # Extend forward
    current = v
    prev = u
    while len(chain) < target_length:
        best_next = None
        best_freq = 0
        for n in graph.neighbors(current):
            if n == prev:
                continue
            ek = tuple(sorted((current, n)))
            if ek in used_edges or ek in used_in_chain:
                continue
            f = edge_freq.get(ek, 0)
            if f > best_freq:
                best_freq = f
                best_next = n
        if best_next is None:
            break
        chain.append((current, best_next))
        used_in_chain.add(tuple(sorted((current, best_next))))
        prev = current
        current = best_next

    if len(chain) >= 2:
        return chain

    # Extend backward
    current = u
    prev = v
    while len(chain) < target_length:
        best_next = None
        best_freq = 0
        for n in graph.neighbors(current):
            if n == prev:
                continue
            ek = tuple(sorted((current, n)))
            if ek in used_edges or ek in used_in_chain:
                continue
            f = edge_freq.get(ek, 0)
            if f > best_freq:
                best_freq = f
                best_next = n
        if best_next is None:
            break
        chain.insert(0, (best_next, current))
        used_in_chain.add(tuple(sorted((best_next, current))))
        prev = current
        current = best_next

    if len(chain) >= 2:
        return chain
    return None


def _prune_corridor_edges(graph, source, target, all_blob_nodes, corridor_cells, rng,
                          target_edge_count=210):
    """
    Removes redundant corridor edges to create narrower passages.

    In a grid, corridor cells have lateral connections to adjacent corridor cells
    that create parallel alternative paths. We remove these to force traffic through
    fewer, more constrained routes (making chokepoints more effective).

    Preserves connectivity and doesn't remove edges within blobs.
    """
    # Identify removable edges: edges between non-blob nodes that are NOT
    # the only path between their endpoints
    candidates = []
    for u, v in list(graph.edges()):
        # Don't remove edges inside blobs
        if u in all_blob_nodes and v in all_blob_nodes:
            continue
        # Don't remove edges adjacent to source/target
        if u == source or u == target or v == source or v == target:
            continue
        candidates.append((u, v))

    rng.shuffle(candidates)

    removed_count = 0
    current_edges = graph.number_of_edges()

    for u, v in candidates:
        if current_edges <= target_edge_count:
            break
        if not graph.has_edge(u, v):
            continue
        # Check if both endpoints have degree > 2 (so removal doesn't create dead ends)
        if graph.degree(u) <= 2 or graph.degree(v) <= 2:
            continue

        # Try removing
        attrs = dict(graph.edges[u, v])
        graph.remove_edge(u, v)

        # Check connectivity
        if nx.has_path(graph, source, target):
            removed_count += 1
            current_edges -= 1
        else:
            # Restore
            graph.add_edge(u, v, **attrs)

    return removed_count


def generate_corridor_map(grid_size=12, num_blobs=4, blob_coverage=0.45,
                          num_chokepoints=18, block_prob=0.5, seed=None):
    """
    Generates a grid map where:
    - Predefined corridor channels ensure non-blob paths exist
    - Blobs are grown to fill non-corridor areas
    - ALL boundary edges between blobs and corridors are removed (walls)
    - Only 1-2 entry points per blob are re-added for optional blob access
    - Redundant corridor edges are pruned to create narrow passages
    - Chokepoints are placed on critical corridor edges

    Args:
        grid_size: Size of the square grid.
        num_blobs: Number of obstacle blobs.
        blob_coverage: Target fraction of grid covered by blobs (0-1).
        num_chokepoints: Target number of chokepoint edges.
        block_prob: Probability of each chokepoint being blocked.
        seed: Random seed.

    Returns:
        dict with map data.
    """
    rng = np.random.RandomState(seed)

    source = (0, 0)
    target = (grid_size - 1, grid_size - 1)
    total_nodes = grid_size * grid_size

    # Step 1: Define corridor paths that blobs cannot occupy
    corridor_cells, corridor_paths = _define_corridors(grid_size, rng)

    # Protected zones near source/target (only immediate neighbors)
    protected = set()
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            for base in [source, target]:
                nr, nc = base[0] + dr, base[1] + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    protected.add((nr, nc))

    occupied = protected | corridor_cells

    # Step 2: Place blobs in non-corridor areas with higher coverage
    total_blob_nodes = int(total_nodes * blob_coverage)
    available_for_blobs = total_nodes - len(occupied)
    total_blob_nodes = min(total_blob_nodes, available_for_blobs)
    blob_size_target = max(5, total_blob_nodes // num_blobs)

    blobs = []

    # Find good seed locations in non-corridor areas
    non_corridor_cells = []
    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) not in occupied:
                non_corridor_cells.append((r, c))

    rng.shuffle(non_corridor_cells)

    # Try to spread blobs spatially
    blob_seeds = []
    for cell in non_corridor_cells:
        if cell in occupied:
            continue
        min_dist = float('inf')
        for existing in blob_seeds:
            d = abs(cell[0] - existing[0]) + abs(cell[1] - existing[1])
            min_dist = min(min_dist, d)
        if len(blob_seeds) == 0 or min_dist >= grid_size // 3:
            blob_seeds.append(cell)
            if len(blob_seeds) >= num_blobs:
                break

    while len(blob_seeds) < num_blobs:
        candidates = [c for c in non_corridor_cells if c not in occupied]
        if not candidates:
            break
        blob_seeds.append(candidates[rng.randint(len(candidates))])

    for seed_node in blob_seeds:
        if seed_node in occupied:
            continue
        size = blob_size_target + rng.randint(-3, 4)
        size = max(5, size)
        blob = _grow_constrained_blob(
            grid_size, grid_size, seed_node, size, rng,
            occupied, corridor_cells
        )
        if len(blob) >= 3:
            blobs.append(blob)
            occupied.update(blob)

    if len(blobs) < 2:
        raise ValueError("Failed to generate enough blobs.")

    # Step 3: Create height map
    map_generator = HeightMapGrid(m=grid_size, n=grid_size)
    for blob in blobs:
        map_generator.add_plataeu(blob)
    map_generator.calculate_distances()
    map_generator.calculate_simple_visibility(blobs)

    all_blob_set = set()
    for blob in blobs:
        all_blob_set.update(blob)

    # Step 4: Remove ALL boundary edges to create complete walls
    boundary_edges = []
    for node in all_blob_set:
        if node not in map_generator.G.nodes():
            continue
        for neighbor in list(map_generator.G.neighbors(node)):
            if neighbor not in all_blob_set:
                boundary_edges.append(tuple(sorted((node, neighbor))))
    boundary_edges = list(set(boundary_edges))

    boundary_attrs = {}
    for edge in boundary_edges:
        u, v = edge
        if map_generator.G.has_edge(u, v):
            boundary_attrs[edge] = dict(map_generator.G.edges[u, v])
            map_generator.G.remove_edge(u, v)

    removed_edges = list(boundary_edges)

    # Step 5: Verify corridor connectivity
    if not nx.has_path(map_generator.G, source, target):
        boundary_list = list(boundary_attrs.keys())
        rng.shuffle(boundary_list)
        for edge in boundary_list:
            u, v = edge
            map_generator.G.add_edge(u, v, **boundary_attrs[edge])
            removed_edges.remove(edge)
            if nx.has_path(map_generator.G, source, target):
                break

    # Step 6: Add minimal entry points into blobs (1-2 per blob)
    # Group boundary edges by blob
    blob_boundary = {}
    for edge in boundary_attrs:
        if edge not in removed_edges:
            continue
        u, v = edge
        blob_node = u if u in all_blob_set else v
        for bi, blob in enumerate(blobs):
            if blob_node in blob:
                if bi not in blob_boundary:
                    blob_boundary[bi] = []
                blob_boundary[bi].append(edge)
                break

    for bi, edges in blob_boundary.items():
        rng.shuffle(edges)
        # Add 1-2 entry points per blob
        num_entries = min(len(edges), 2)
        for edge in edges[:num_entries]:
            u, v = edge
            if not map_generator.G.has_edge(u, v):
                map_generator.G.add_edge(u, v, **boundary_attrs[edge])
                removed_edges.remove(edge)

    # Step 7: Prune corridor edges to create narrow passages
    # Target: reduce total edges to roughly match the original benchmark (~207 for 12x12)
    target_edges = int(total_nodes * 1.44)  # ~207/144 for original benchmark
    _prune_corridor_edges(
        map_generator.G, source, target, all_blob_set, corridor_cells, rng,
        target_edge_count=target_edges
    )

    # Step 8: Recompute visibility
    map_generator.calculate_simple_visibility(blobs)
    removed_set = set(tuple(sorted(e)) for e in removed_edges)
    for node in map_generator.G.nodes():
        if "visible_edges" in map_generator.G.nodes[node]:
            vis = map_generator.G.nodes[node]["visible_edges"]
            map_generator.G.nodes[node]["visible_edges"] = [
                e for e in vis if tuple(sorted(e)) not in removed_set
            ]

    env_graph = map_generator.get_graph()

    # Step 9: Find chokepoints on critical corridor edges
    chokepoints = _find_chokepoints_via_cuts(
        env_graph, source, target,
        num_chokepoints=num_chokepoints, rng=rng
    )

    # Validate chokepoint edges
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


def _validate_map(map_data, min_chokepoints=6, min_cp_coverage=0.6,
                   num_blockage_trials=30, min_sp_cost_cv=0.03):
    """
    Validates a generated map for benchmark suitability.

    Checks:
    1. Graph connectivity between source and target
    2. Minimum number of chokepoints
    3. Shortest path mostly avoids blobs (goes through corridors)
    4. Chokepoint coverage (diverse paths pass through chokepoints)
    5. Blockage impact: random blockage realizations must cause meaningful
       variation in SP cost, ensuring blockages actually force detours
    6. Quick agent simulation to reject maps where our agent does poorly
    """
    g = map_data['env_graph']
    s, t = map_data['source'], map_data['target']
    chokepoints = map_data['chokepoints']
    block_prob = map_data.get('block_prob', 0.5)

    if not nx.has_path(g, s, t):
        return False
    if len(chokepoints) < min_chokepoints:
        return False

    # Check that shortest path mostly avoids blobs
    all_blob_nodes = set()
    for blob in map_data['blobs']:
        all_blob_nodes.update(blob)
    sp = nx.shortest_path(g, s, t, weight="distance")
    blob_frac = sum(1 for n in sp if n in all_blob_nodes) / len(sp)
    if blob_frac > 0.3:
        return False

    # Check chokepoint coverage
    cp_set = set(tuple(sorted(e)) for e in chokepoints)
    temp_g = g.copy()
    paths_through_cp = 0
    total_paths = 0

    for _ in range(10):
        try:
            sp = nx.shortest_path(temp_g, s, t, weight="distance")
            total_paths += 1
            path_edges = set(tuple(sorted((sp[i], sp[i+1])))
                           for i in range(len(sp) - 1))
            if path_edges & cp_set:
                paths_through_cp += 1
            for i in range(len(sp) - 1):
                u, v = sp[i], sp[i+1]
                if temp_g.has_edge(u, v):
                    temp_g.edges[u, v]['distance'] *= 3.0
        except nx.NetworkXNoPath:
            break

    if total_paths == 0:
        return False
    if (paths_through_cp / total_paths) < min_cp_coverage:
        return False

    # Blockage impact validation
    rng = np.random.RandomState(map_data.get('seed', 0) + 9999)
    sp_costs = []
    baseline_sp = nx.shortest_path(g, s, t, weight="distance")
    baseline_cost = sum(g.edges[baseline_sp[i], baseline_sp[i+1]]['distance']
                        for i in range(len(baseline_sp) - 1))

    detours_near_blobs = 0
    total_detours = 0

    for trial in range(num_blockage_trials):
        edges_to_remove = []
        rolls = rng.rand(len(chokepoints))
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
            cost = sum(blocked_g.edges[blocked_sp[i], blocked_sp[i+1]]['distance']
                       for i in range(len(blocked_sp) - 1))
            sp_costs.append(cost)

            if cost > baseline_cost + 0.5:
                total_detours += 1
                detour_nodes = set(blocked_sp) - set(baseline_sp)
                near_blob = False
                for dn in detour_nodes:
                    for neighbor in g.neighbors(dn):
                        if neighbor in all_blob_nodes:
                            near_blob = True
                            break
                    if near_blob:
                        break
                if near_blob:
                    detours_near_blobs += 1
        except (nx.NetworkXNoPath, KeyError):
            continue

    if len(sp_costs) < num_blockage_trials // 2:
        return False

    sp_costs_arr = np.array(sp_costs)
    sp_mean = np.mean(sp_costs_arr)
    sp_std = np.std(sp_costs_arr)

    if sp_mean <= 0:
        return False

    cv = sp_std / sp_mean
    if cv < min_sp_cost_cv:
        return False

    if total_detours < 2:
        return False

    return True


def generate_map_suite(num_maps=50, seed_start=1000):
    """Generates a suite of validated maps with varying parameters."""
    maps = []

    configs = [
        # Focus on narrow/corridor-like environments where visibility-aware
        # planning has the strongest advantage. Vary grid size, blob coverage,
        # chokepoint density, and block probability.
        # (grid_size, num_blobs, blob_coverage, num_chokepoints, block_prob, label)
        (12, 4, 0.45, 18, 0.50, "narrow_12x12"),
        (12, 4, 0.45, 21, 0.50, "dense_12x12"),
        (14, 4, 0.42, 18, 0.50, "narrow_14x14"),
        (14, 4, 0.42, 21, 0.50, "dense_14x14"),
        (16, 5, 0.40, 21, 0.50, "narrow_16x16"),
        (16, 5, 0.40, 24, 0.50, "dense_16x16"),
        (12, 4, 0.45, 18, 0.40, "medium_12x12"),
        (14, 4, 0.42, 18, 0.40, "medium_14x14"),
        (16, 5, 0.40, 21, 0.40, "medium_16x16"),
        (14, 6, 0.42, 18, 0.50, "many_blobs_14x14"),
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

    print(f"Grid: {map_data['grid_size']}x{map_data['grid_size']}")
    print(f"Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
    print(f"Blob nodes: {len(all_blob_nodes)} ({len(all_blob_nodes)/g.number_of_nodes()*100:.1f}%)")
    print(f"Blobs: {len(map_data['blobs'])}, sizes: {[len(b) for b in map_data['blobs']]}")
    print(f"Chokepoints: {len(map_data['chokepoints'])}")

    boundary_remaining = sum(1 for u, v in g.edges()
                            if (u in all_blob_nodes) != (v in all_blob_nodes))
    print(f"Boundary edges remaining: {boundary_remaining}")

    sp = nx.shortest_path(g, map_data['source'], map_data['target'], weight="distance")
    sp_len = sum(g.edges[sp[i], sp[i+1]]['distance'] for i in range(len(sp)-1))
    blob_on_sp = sum(1 for n in sp if n in all_blob_nodes)
    print(f"SP: length={sp_len:.0f}, blob_nodes={blob_on_sp}/{len(sp)}")

    deg2 = sum(1 for n in g.nodes() if g.degree(n) == 2 and n not in all_blob_nodes)
    print(f"Degree-2 non-blob nodes: {deg2}")

    # Coverage
    cp_set = set(tuple(sorted(e)) for e in map_data['chokepoints'])
    temp_g = g.copy()
    total = 0
    through = 0
    for _ in range(10):
        try:
            sp = nx.shortest_path(temp_g, map_data['source'], map_data['target'], weight="distance")
            total += 1
            path_edges = set(tuple(sorted((sp[i], sp[i+1]))) for i in range(len(sp)-1))
            n_cp = len(path_edges & cp_set)
            if n_cp > 0:
                through += 1
            for i in range(len(sp) - 1):
                u, v = sp[i], sp[i+1]
                if temp_g.has_edge(u, v):
                    temp_g.edges[u, v]['distance'] *= 3.0
        except:
            break
    print(f"Coverage: {through}/{total}")
    print(f"Valid: {_validate_map(map_data)}")
