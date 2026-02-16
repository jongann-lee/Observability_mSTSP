import numpy as np
import networkx as nx
from tqdm import tqdm
import time
import random
import itertools

from Graph_Generation.height_graph_generation import HeightMapGrid
from Graph_Generation.target_graph import create_fully_connected_target_graph
from Graph_Generation.edge_block_generation import block_specific_edges
from Single_Agent.repeated_topk import RepeatedTopK
from Single_Agent.RPP import ZeroCostRPP

# Settings
edge_block_prob = float(0.5)
num_runs = 1000

use_shortest_path_agent = False
use_our_agent = True
use_RPP_agent = False

# ============================================================
# MAP DESIGN: "Improved Organic Mountains with Corridor Divergence"
# 12x12 grid, source (0,0), target (11,11)
#
# 4 organic mountains in each quadrant with 2-4 access points each.
# Mountain2 has blob nodes at x=5 (y=4-7) for extended visibility.
# Multiple corridors: perimeter (y=0→x=11), central (x=6), and top (y=11).
#
# SP agent takes perimeter route (cost 22.0).
# RepeatedTopK takes central x=6 corridor at ratio=1,3 (cost 22.0),
# or extended visibility route at ratio=5,10 (cost 26.0).
# ============================================================

# Mountain 1 (top-left, 8 nodes): compact L-shape
mountain1 = [(0,11), (1,11), (2,11),
             (0,10), (1,10), (2,10),
             (0,9),  (1,9)]

# Mountain 2 (center-left, 12 nodes): rectangular blob with key x=5 nodes
mountain2 = [(3,4), (4,4), (5,4),
             (3,5), (4,5), (5,5),
             (3,6), (4,6), (5,6),
             (3,7), (4,7), (5,7)]

# Mountain 3 (bottom-right, 13 nodes): organic slab tapering south
mountain3 = [(7,1), (8,1), (9,1),
             (7,2), (8,2), (9,2), (10,2),
             (7,3), (8,3), (9,3), (10,3),
             (8,4), (9,4)]

# Mountain 4 (top-right, 13 nodes): organic blob tapering northeast
mountain4 = [(7,8),  (8,8),  (9,8),  (10,8),
             (7,9),  (8,9),  (9,9),  (10,9),
             (8,10), (9,10), (10,10),
             (9,11), (10,11)]

set_of_blobs = [mountain1, mountain2, mountain3, mountain4]

map_generator = HeightMapGrid(m=12, n=12)
map_generator.add_plataeu(mountain1)
map_generator.add_plataeu(mountain2)
map_generator.add_plataeu(mountain3)
map_generator.add_plataeu(mountain4)
map_generator.calculate_distances()
map_generator.calculate_simple_visibility(set_of_blobs)

# Cliff edges around mountains (create impassable walls with access gaps)

# Mountain 1 (top-left, 8 nodes) - 2 access points
# Access: (1,9)→(1,8) south, (2,10)→(3,10) east
edge_list_1 = [
    ((0,8), (0,9)),    # south-west wall
    ((2,11), (3,11)),  # east-top wall
    ((2,9), (3,9)),    # east-bottom wall
]

# Mountain 2 (center, 12 nodes) - 3 access points
# Access: (3,4)→(3,3) south, (3,6)→(2,6) west, (5,7)→(5,8) north
edge_list_2 = [
    # East wall (x=5→x=6) - sealed to create corridor separation
    ((5,4), (6,4)), ((5,5), (6,5)), ((5,6), (6,6)), ((5,7), (6,7)),
    # North wall (leave (5,7)→(5,8) open for north exit)
    ((3,7), (3,8)), ((4,7), (4,8)),
    # South wall (leave (3,4)→(3,3) open)
    ((4,4), (4,3)), ((5,4), (5,3)),
    # West wall (leave (3,6)→(2,6) open)
    ((3,4), (2,4)), ((3,5), (2,5)), ((3,7), (2,7)),
]

# Mountain 3 (bottom-right, 13 nodes) - 3 access points
# Access: (7,1)→(7,0) south, (10,3)→(11,3) east, (8,4)→(8,5) north
edge_list_3 = [
    # West wall (x=6→x=7)
    ((6,1), (7,1)), ((6,2), (7,2)), ((6,3), (7,3)),
    # South wall (leave (7,1)→(7,0) open)
    ((8,1), (8,0)), ((9,1), (9,0)), ((9,1), (10,1)),
    # East wall (leave (10,3)→(11,3) open)
    ((10,2), (11,2)), ((10,3), (10,4)),
    # North wall (leave (8,4)→(8,5) open)
    ((9,4), (9,5)), ((8,4), (7,4)),
]

# Mountain 4 (top-right, 13 nodes) - 4 access points
# Access: (7,8)→(6,8) west, (10,9)→(11,9) east, (9,11)→(8,11) north, (10,11)→(11,11) east-top
edge_list_4 = [
    # West wall (leave (7,8)→(6,8) open)
    ((7,9), (6,9)), ((8,10), (7,10)), ((7,9), (7,10)),
    # South wall
    ((7,8), (7,7)), ((8,8), (8,7)), ((9,8), (9,7)), ((10,8), (10,7)),
    # East wall (leave (10,9)→(11,9) and (10,11)→(11,11) open)
    ((10,8), (11,8)), ((10,10), (11,10)),
    # North wall (leave (9,11)→(8,11) open)
    ((8,10), (8,11)),
]

edge_list = edge_list_1 + edge_list_2 + edge_list_3 + edge_list_4
map_generator.remove_edges(edge_list)

env_graph = map_generator.get_graph()

target_recursion = 4
target_num_obstacles = 4
target_obstacle_hop = 1

target_graph = create_fully_connected_target_graph(env_graph, recursions=target_recursion,
                                                    num_obstacles=target_num_obstacles,
                                                    obstacle_hop=target_obstacle_hop)


# Define the chokepoints (the edges that can be blocked)
# Non-bridge edges spread across 6 corridor groups.

chokepoints_list = [
    # Group 1: Central corridor x=6, lower section
    ((6,1), (6,2)), ((6,2), (6,3)), ((6,3), (6,4)),
    # Group 2: Central corridor x=6, upper section
    ((6,7), (6,8)), ((6,8), (6,9)), ((6,9), (6,10)),
    # Group 3: Top corridor y=11
    ((3,11), (4,11)), ((4,11), (5,11)), ((5,11), (6,11)),
    # Group 4: East corridor x=11
    ((11,4), (11,5)), ((11,5), (11,6)), ((11,6), (11,7)),
    # Group 5: Bottom connector y=0
    ((4,0), (5,0)), ((5,0), (6,0)),
    # Group 6: South corridor y=0 (right side)
    ((8,0), (9,0)), ((9,0), (10,0)),
]

# Pre calculate shortest path and the Hamiltonian target path(trivial for now)
path_generator = RepeatedTopK(reward_ratio = 1.0, env_graph=env_graph, target_graph=target_graph,
                              sample_recursion=4, sample_num_obstacle=4, sample_obstacle_hop=1)

shortest_path = []
hamiltonian_target_path = path_generator.generate_Hamiltonian_path()
for i in range(len(hamiltonian_target_path) - 1):
    begin_node = hamiltonian_target_path[i]
    end_node = hamiltonian_target_path[i + 1]

    section_path = nx.shortest_path(env_graph, source=begin_node, target=end_node, weight="distance")

    # Append the section best path, avoiding duplication of nodes at the end
    if len(shortest_path) > 0 and shortest_path[-1] == section_path[0]:
        shortest_path.extend(section_path[1:])
    else:
        shortest_path.extend(section_path)

path_length_list = []
runtimes = []


# Main repeating run loop
for run_idx in tqdm(range(num_runs)):

    # Set the seed here so that each test gets the same samples of the edge block distribution
    np.random.seed(42+run_idx)
    random.seed(42+run_idx)

    start_time = time.time()

    # Block edges
    edges_to_remove = []
    RNG = np.random.rand(len(chokepoints_list))
    for i, (u, v) in enumerate(chokepoints_list):
        if RNG[i] < edge_block_prob:
            edges_to_remove.append((u, v))


    # Create the blocked env graph (real env) and remove the blocked edges
    blocked_env_graph = block_specific_edges(env_graph, edges_to_remove)
    blocked_env_graph.remove_edges_from(edges_to_remove)

    # Skip run if the goal is unreachable
    if not nx.has_path(blocked_env_graph, (0,0), (11,11)):
        continue

    # Update the visible_edges for all nodes to reflect the removed edges
    for node in blocked_env_graph.nodes():
        if "visible_edges" in blocked_env_graph.nodes[node]:
            current_visible = blocked_env_graph.nodes[node]["visible_edges"]
            updated_visible = set(current_visible) - set(edges_to_remove)
            blocked_env_graph.nodes[node]["visible_edges"] = list(updated_visible)

    # Get the path taken by the shortest path agent
    if use_shortest_path_agent:
        path_1 = shortest_path.copy()
        target_nodes = hamiltonian_target_path.copy()
        env_graph1 = env_graph.copy()

        current_node = path_1[0]
        next_target_index = 1
        index = 0
        total_travel_distance = 0.0

        while index < len(path_1) - 1:
            next_node = path_1[index + 1]

            if current_node == target_nodes[next_target_index]:
                next_target_index += 1

            observable_edges = set(blocked_env_graph.nodes[current_node]["visible_edges"])
            assumed_observable_edges = set(env_graph1.nodes[current_node]["visible_edges"])
            blocked_edges = assumed_observable_edges - observable_edges

            if len(blocked_edges) > 0:
                for edge in blocked_edges:
                    u, v = edge
                    if env_graph1.has_edge(u, v):
                        env_graph1.remove_edge(u, v)
                for node in env_graph1.nodes():
                    if "visible_edges" in env_graph1.nodes[node]:
                        current_visible = env_graph1.nodes[node]["visible_edges"]
                        updated_visible = set(current_visible) - blocked_edges
                        env_graph1.nodes[node]["visible_edges"] = list(updated_visible)

                path_edges = [(path_1[i], path_1[i+1]) for i in range(index, len(path_1) - 1)]

                blocked_edges_both_directions = set()
                for u, v in blocked_edges:
                    blocked_edges_both_directions.add((u, v))
                    blocked_edges_both_directions.add((v, u))

                if any(edge in blocked_edges_both_directions for edge in path_edges):
                    remaining_targets = target_nodes[next_target_index:]
                    new_path = [current_node]

                    for target in remaining_targets:
                        segment = nx.shortest_path(env_graph1, source=new_path[-1], target=target, weight="distance")
                        new_path.extend(segment[1:])

                    path_1 = path_1[:index + 1] + new_path[1:]

            next_node = path_1[index + 1]

            total_travel_distance += env_graph.edges[current_node, next_node]["distance"]
            current_node = next_node
            index += 1

        path_length_list.append(total_travel_distance)

    elif use_our_agent:
        env_graph2 = env_graph.copy()
        path2_generator = RepeatedTopK(reward_ratio = 3.0, env_graph=env_graph2, target_graph=target_graph,
                                       sample_recursion=4, sample_num_obstacle=4, sample_obstacle_hop=1)

        path_2 = path2_generator.find_best_path()
        target_nodes = hamiltonian_target_path.copy()

        current_node = path_2[0]
        next_target_index = 1
        index = 0
        total_travel_distance = 0.0

        while index < len(path_2) - 1:
            next_node = path_2[index + 1]

            if current_node == target_nodes[next_target_index]:
                next_target_index += 1

            observable_edges = set(blocked_env_graph.nodes[current_node]["visible_edges"])
            assumed_observable_edges = set(env_graph2.nodes[current_node]["visible_edges"])
            blocked_edges = assumed_observable_edges - observable_edges

            # MARK OBSERVED EDGES AS SEEN
            for edge in assumed_observable_edges:
                if env_graph2.has_edge(*edge):
                    env_graph2.edges[edge]["observed_edge"] = True

            if len(blocked_edges) > 0:
                for edge in blocked_edges:
                    u, v = edge
                    if env_graph2.has_edge(u, v):
                        env_graph2.remove_edge(u, v)
                for node in env_graph2.nodes():
                    if "visible_edges" in env_graph2.nodes[node]:
                        current_visible = env_graph2.nodes[node]["visible_edges"]
                        updated_visible = set(current_visible) - blocked_edges
                        env_graph2.nodes[node]["visible_edges"] = list(updated_visible)

                path_edges = [(path_2[i], path_2[i+1]) for i in range(index, len(path_2) - 1)]

                blocked_edges_both_directions = set()
                for u, v in blocked_edges:
                    blocked_edges_both_directions.add((u, v))
                    blocked_edges_both_directions.add((v, u))

                if any(edge in blocked_edges_both_directions for edge in path_edges):
                    remaining_targets = target_nodes[next_target_index:]
                    new_path = [current_node]

                    for target in remaining_targets:
                        segment = path2_generator.alternate_path_online(new_path[-1], target)
                        new_path.extend(segment[1:])

                    path_2 = path_2[:index + 1] + new_path[1:]

            next_node = path_2[index + 1]

            total_travel_distance += env_graph.edges[current_node, next_node]["distance"]
            current_node = next_node
            index += 1

        path_length_list.append(total_travel_distance)

    elif use_RPP_agent:
        env_graph4 = env_graph.copy()

        realizations = []
        chokepoints = chokepoints_list

        source_node = (0,0)
        target_node = (11,11)
        edge_block_prob = edge_block_prob

        for bits in itertools.product([0, 1], repeat=len(chokepoints)):
            blocked_in_this_world = [chokepoints[i] for i, bit in enumerate(bits) if bit == 1]

            prob = 1.0
            for bit in bits:
                prob *= edge_block_prob if bit == 1 else (1 - edge_block_prob)

            all_edges = set(env_graph.edges())
            free_edges = all_edges - set(blocked_in_this_world)

            realizations.append({
                'prob': prob,
                'edges': free_edges,
                'blocked': set(blocked_in_this_world)
            })


        path4_generator = ZeroCostRPP(env_graph=env_graph4,
                                    realizations=realizations,
                                    start_node=source_node,
                                    goal_node=target_node)

        policy_tree = path4_generator.plan()

        current_belief_indices = tuple(range(len(realizations)))
        current_tree_node = (current_belief_indices, source_node)
        current_node = source_node

        agent4_travel_distance = 0.0

        while True:
            node_data = policy_tree.nodes.get(current_tree_node, {})

            if node_data.get('type') == 'terminal_no_goal':
                break

            if current_node == target_node:
                break

            successors = list(policy_tree.successors(current_tree_node))
            if not successors:
                break

            next_tree_state = successors[0]
            target_env_vertex = next_tree_state[1]

            leg_path = nx.shortest_path(blocked_env_graph, current_node, target_env_vertex, weight="distance")

            for next_step in leg_path[1:]:
                distance = blocked_env_graph.edges[current_node, next_step]["distance"]
                agent4_travel_distance += distance
                current_node = next_step

            if current_node != target_node:
                visible_here = env_graph.nodes[current_node].get("visible_edges", [])
                actual_outcome = tuple(sorted(
                    [e for e in visible_here if blocked_env_graph.has_edge(*e)],
                    key=lambda x: (x[0], x[1])
                ))

                found_next_belief = False
                for potential_next_belief in policy_tree.successors(next_tree_state):
                    edge_data = policy_tree.get_edge_data(next_tree_state, potential_next_belief)
                    if edge_data.get('observation_result') == actual_outcome:
                        current_tree_node = potential_next_belief
                        found_next_belief = True
                        break

                if not found_next_belief:
                    break
            else:
                break

        path_length_list.append(agent4_travel_distance)


    end_time = time.time()
    runtimes.append(end_time - start_time)

if path_length_list:
    mean_cost = np.mean(path_length_list)
    variance_cost = np.var(path_length_list)
    std_dev = np.std(path_length_list)
    avg_runtime = np.mean(runtimes)

    if use_our_agent:
        print("\n" + "="*30)
        print(f"MODEL PARAMETERS")
        print(f"Edge Recursion:     {target_recursion:d}")
        print(f"Edge num obstacles: {target_num_obstacles:d}")
        print(f"Edge obstacle hop:  {target_obstacle_hop:d}")
        print(f"Plan recursion      {path2_generator.recursion:d}")
        print(f"Plan num obstacles: {path2_generator.num_obstacle:d}")
        print(f"Plan obstacle hop:  {path2_generator.obstacle_hop:d}")
        print(f"Reward ratio:       {path2_generator.reward_ratio:.2f}")

    print("-" * 30)
    print(f"RESULTS FOR {len(path_length_list)} SUCCESSFUL RUNS")
    print("-" * 30)
    print(f"Mean Path Cost:     {mean_cost:.2f}")
    print(f"Variance:           {variance_cost:.2f}")
    print(f"Std Deviation:      {std_dev:.2f}")
    print(f"Avg Runtime/Run:    {avg_runtime:.4f}s")
    print("="*30)
