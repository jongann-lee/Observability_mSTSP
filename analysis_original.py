"""Analyze the original benchmark.py map to understand what makes it good."""
import numpy as np
import networkx as nx
from Graph_Generation.height_graph_generation import HeightMapGrid
from Graph_Generation.target_graph import create_fully_connected_target_graph, stochastic_accumulated_blockage_path
from Single_Agent.repeated_topk import RepeatedTopK

# Original map
mountain1 = [(0,11), (1,11), (2,11), (3,11),
             (0,10), (1,10), (2,10), (3,10),
             (0,9),  (1,9),  (2,9),
             (0,8), (1,8)]
mountain2 = [(3,7), (4,7), (5,7),
             (1,6), (2,6), (3,6), (4,6), (5,6),
             (1,5), (2,5), (3,5), (4,5),
             (1,4), (2,4), (3,4), (4,4),
             (3,3), (4,3)]
mountain3 = [(7,5),
         (6,4), (7,4), (8,4), (9,4), (10,4),
         (6,3), (7,3), (8,3), (9,3), (10,3),
         (6,2), (7,2), (8,2), (9,2),
         (6,1), (7,1)]
mountain4 = [(6,10), (7,10), (8,10), (9,10), (10,10),
             (6,9), (7,9), (8,9), (9,9), (10,9),
             (7,8), (8,8), (9,8), (10,8),
             (9,7), (10,7),
             (9,6), (10,6)]
set_of_blobs = [mountain1, mountain2, mountain3, mountain4]
all_mountain_nodes = set(mountain1 + mountain2 + mountain3 + mountain4)

map_gen = HeightMapGrid(m=12, n=12)
map_gen.add_plataeu(mountain1); map_gen.add_plataeu(mountain2)
map_gen.add_plataeu(mountain3); map_gen.add_plataeu(mountain4)
map_gen.calculate_distances()
map_gen.calculate_simple_visibility(set_of_blobs)

edge_list_1 = [((0,7), (0,8)), ((1,8), (2,8)), ((2,8), (2,9)), ((2,9), (3,9)),
               ((3,9), (3,10)), ((3,11), (4,11))]
edge_list_2 = [((3,7), (3,8)), ((4,7), (4,8)), ((5,7), (5,8)),
               ((5,7), (6,7)), ((5,6), (6,6)), ((5,5), (5,6)),
                ((4,5), (5,5)), ((4,4), (5,4)), ((4,3), (5,3)),
                ((4,2), (4,3)), ((3,2), (3,3)), ((2,3), (2,4)),
                ((1,3), (1,4)), ((0,4), (1,4)), ((0,5), (1,5)),
                ((0,6), (1,6)), ((1,6), (1,7)), ((2,6), (2,7))]
edge_list_3 = [((7,5), (8,5)), ((8,4), (8,5)), ((9,4), (9,5)),
               ((10,4), (10,5)), ((10,4), (11,4)), ((10,3), (11,3)),
               ((10,2), (10,3)), ((8,1), (8,2)),
                ((7,0), (7,1)), ((9,1), (9,2)),
               ((5,1), (6,1)), ((5,2), (6,2)), ((5,3), (6,3)),
               ((5,4), (6,4)), ((6,4), (6,5)), ((6,5), (7,5))]
edge_list_4 = [((7,10), (7,11)), ((8,10), (8,11)), ((9,10), (9,11)),
               ((10,10), (10,11)), ((10,10), (11,10)), ((10,9), (11,9)),
               ((10,8), (11,8)), ((10,7), (11,7)), ((10,5), (10,6)),
               ((9,5), (9,6)), ((8,6), (9,6)), ((8,7), (8,8)),
               ((7,7), (7,8)), ((6,8), (7,8)), ((6,8), (6,9)),
               ((5,9), (6,9)), ((5,10), (6,10))]
edge_list = edge_list_1 + edge_list_2 + edge_list_3 + edge_list_4
map_gen.remove_edges(edge_list)
env_graph = map_gen.get_graph()

# 1. Print ASCII map showing heights and connectivity
print("=== MAP LAYOUT (height=0 is '.', height=1 is '#') ===")
for y in range(11, -1, -1):
    row = f"y={y:2d} "
    for x in range(12):
        h = env_graph.nodes[(x,y)].get('height', 0)
        if (x,y) == (0,0): row += "S"
        elif (x,y) == (11,11): row += "T"
        elif h > 0: row += "#"
        else: row += "."
    print(row)
print("     " + "".join(f"{x}" for x in range(12)))

# 2. How many cliff-edge entries does each mountain have?
print("\n=== MOUNTAIN ACCESS POINTS (non-cliff boundary edges) ===")
for name, blob in [("m1", mountain1), ("m2", mountain2), ("m3", mountain3), ("m4", mountain4)]:
    blob_set = set(blob)
    entries = []
    for node in blob:
        for nb in env_graph.neighbors(node):
            if nb not in blob_set:
                entries.append((node, nb))
    print(f"  {name}: {len(entries)} access points")
    for src, dst in entries:
        h_src = env_graph.nodes[src]['height']
        h_dst = env_graph.nodes[dst]['height']
        dist = env_graph.edges[src, dst]['distance']
        print(f"    ({src})->({dst}): h={h_src}->{h_dst}, dist={dist:.1f}")

# 3. Bridges
bridges = list(nx.bridges(env_graph))
bridge_set = set(bridges) | set((v,u) for u,v in bridges)
print(f"\n=== BRIDGES: {len(bridges)} ===")
for b in sorted(bridges):
    print(f"  {b}")

# 4. Shortest path
sp = nx.shortest_path(env_graph, (0,0), (11,11), weight="distance")
sp_len = sum(env_graph.edges[sp[i], sp[i+1]]["distance"] for i in range(len(sp)-1))
print(f"\n=== SHORTEST PATH: len={sp_len:.2f} ===")
print(f"  {sp}")

# 5. Alternative routes - force through different corridors
print("\n=== ALTERNATIVE ROUTES ===")
# Try going through different x-columns at y=6
for via_x in [0, 2, 5, 6, 8, 11]:
    via = (via_x, 6)
    if via in env_graph:
        try:
            p1 = nx.shortest_path(env_graph, (0,0), via, weight="distance")
            p2 = nx.shortest_path(env_graph, via, (11,11), weight="distance")
            full = p1 + p2[1:]
            total = sum(env_graph.edges[full[i], full[i+1]]["distance"] for i in range(len(full)-1))
            print(f"  Via ({via_x},6): len={total:.2f}")
        except:
            print(f"  Via ({via_x},6): no path")

# 6. Diverse paths analysis
target_graph = create_fully_connected_target_graph(env_graph, recursions=4, num_obstacles=4, obstacle_hop=1)
diverse = stochastic_accumulated_blockage_path(env_graph, (0,0), (11,11), recursions=4, num_obstacles_per_path=4, obstacle_hop=1)
print(f"\n=== DIVERSE PATHS: {len(diverse)} total ===")
lengths = {}
for path, depth in diverse:
    plen = sum(env_graph.edges[path[j], path[j+1]]["distance"] for j in range(len(path)-1))
    if plen not in lengths: lengths[plen] = 0
    lengths[plen] += 1
for l in sorted(lengths.keys()):
    print(f"  Length {l:.1f}: {lengths[l]} paths")

# 7. Check path at each y-value to see which column the diverse paths use
print(f"\n=== DIVERSE PATH CORRIDORS (x-column at y=6) ===")
for path, depth in diverse[:20]:
    plen = sum(env_graph.edges[path[j], path[j+1]]["distance"] for j in range(len(path)-1))
    # Find where path crosses y=6
    y6_nodes = [n for n in path if n[1] == 6]
    y6_x = [n[0] for n in y6_nodes] if y6_nodes else []
    print(f"  d={depth} len={plen:.1f} x@y6={y6_x}")

# 8. SP vs Our agent initial paths
gen = RepeatedTopK(reward_ratio=1.0, env_graph=env_graph, target_graph=target_graph,
                   sample_recursion=4, sample_num_obstacle=4, sample_obstacle_hop=1)
ham = gen.generate_Hamiltonian_path()

shortest_path = []
for i in range(len(ham)-1):
    section = nx.shortest_path(env_graph, ham[i], ham[i+1], weight="distance")
    if shortest_path and shortest_path[-1] == section[0]:
        shortest_path.extend(section[1:])
    else:
        shortest_path.extend(section)

print(f"\n=== SP AGENT PATH ===")
print(f"  {shortest_path}")

for ratio in [3.0, 5.0, 10.0]:
    env_copy = env_graph.copy()
    g = RepeatedTopK(reward_ratio=ratio, env_graph=env_copy, target_graph=target_graph,
                     sample_recursion=4, sample_num_obstacle=4, sample_obstacle_hop=1)
    p = g.find_best_path()
    total = sum(env_graph.edges[p[i], p[i+1]]["distance"] for i in range(len(p)-1))
    y6_nodes = [n for n in p if n[1] == 6]
    diverges = "DIVERGE" if shortest_path != p else "SAME"
    print(f"\n=== RATIO={ratio} len={total:.2f} [{diverges}] ===")
    print(f"  {p}")
    if diverges == "DIVERGE":
        for k in range(min(len(shortest_path), len(p))):
            if shortest_path[k] != p[k]:
                print(f"  Divergence at step {k}: SP->{shortest_path[k]}, ours->{p[k]}")
                break

# 9. Chokepoints analysis
chokepoints_list = [((7,11), (8,11)), ((8,11), (9,11)), ((9,11), (10,11)),
                    ((11,7), (11,8)), ((11,8), (11,9)), ((11,9), (11,10)),
                    ((8,5), (8,6)), ((8,5), (9,5)), ((9,5), (10,5)),
                    ((0,4), (0,5)), ((0,5), (0,6)), ((0,6), (0,7)),
                    ((5,3), (5,4)), ((5,4), (5,5)), ((5,5), (6,5)),
                    ((11,2), (11,3)), ((11,3), (11,4)), ((11,4), (11,5))]

print(f"\n=== CHOKEPOINTS ===")
for cp in chokepoints_list:
    u, v = cp
    is_bridge = cp in bridge_set or (v,u) in bridge_set
    nu = env_graph.edges[u,v].get('num_used', 0) if env_graph.has_edge(u,v) else 0
    on_sp = any((sp[i],sp[i+1]) == cp or (sp[i+1],sp[i]) == cp for i in range(len(sp)-1))
    print(f"  {cp}: bridge={is_bridge}, num_used={nu:.3f}, on_SP={on_sp}")
