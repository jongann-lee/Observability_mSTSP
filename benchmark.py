import numpy as np
import networkx as nx
from tqdm import tqdm
import time

from Graph_Generation.simple_graph_generation import create_occupancy_grid
from Graph_Generation.target_graph import create_fully_connected_target_graph
from Graph_Generation.visibility import hill_visibility
from Graph_Generation.edge_block_generation import block_specific_edges
from Single_Agent.repeated_topk import RepeatedTopK
from Single_Agent.VOI import VOITrajectoryManager

# Settings
np.random.seed(42)
edge_block_prob = 0.75
num_runs = 100

use_shortest_path_agent = True

# Generate the graph (same as in single_agent_blocked.ipynb)

# We will remove three blobs
blob1 = [(1,12), (2,12), (2,11), (3,11), (3,10), (4,10)]
blob2 = [(6,8), (7,7), (6,9), (7,8), (8,7), (8,8), (7,9), (9,6)]
blob3 = [(7,1), (7,2), (8,1), (8,2), (8,3), (8,4), (9,2), (9,3), (9,4)]
blob4 = [(7,12), (8,12), (7,13), (8,13)]
blob5 = [(4,0), (5,0)]
sef_of_blobs = blob1 + blob2 + blob3 + blob4 + blob5

# We also define the hill nodes
hill_nodes = [(2,2), (2,3), (3,2), (3,3)]


env_graph = create_occupancy_grid(m=14, n=14, nodes_to_remove=sef_of_blobs)
env_graph = hill_visibility(env_graph, hill_nodes)
target_graph = create_fully_connected_target_graph(env_graph)

# Define the chokepoints (the edges that can be blocked)
chokepoints_list = [((1,13), (2,13)),
                    ((5,9), (5,10)),
                    ((8,5), (9,5)),
                    ((7,0), (8,0))]

# Pre calculate shortest path and the Hamiltonian target path(trivial for now)
path_generator = RepeatedTopK(reward_ratio = 1.0, env_graph=env_graph, target_graph=target_graph)

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

    start_time = time.time()

    # Block edges
    edges_to_remove = []
    RNG = np.random.rand(len(chokepoints_list)) # if below 0.75 then edge is blocked
    for i, (u, v) in enumerate(chokepoints_list):
        if RNG[i] < edge_block_prob:
            edges_to_remove.append((u, v))
            print(f" Blocking edge{i}")
    # Skip run if all chokepoints are blocked (ensuring graph connectivity)
    if len(edges_to_remove) == len(chokepoints_list):
        continue

    # Create the blocked env graph (real env) and remove the blocked edges
    blocked_env_graph = block_specific_edges(env_graph, edges_to_remove)
    blocked_env_graph.remove_edges_from(edges_to_remove)

    # Update the visible_edges for all nodes to reflect the removed edges
    for node in blocked_env_graph.nodes():
        if "visible_edges" in blocked_env_graph.nodes[node]:
            current_visible = blocked_env_graph.nodes[node]["visible_edges"]
            # Remove any blocked edges from the visible_edges set
            updated_visible = set(current_visible) - set(edges_to_remove)
            blocked_env_graph.nodes[node]["visible_edges"] = list(updated_visible)

    # Get the path taken by the shortest path agent
    if use_shortest_path_agent:
        path_1 = shortest_path.copy() # Start with the shortest path
        target_nodes = hamiltonian_target_path.copy()  # All target nodes
        env_graph1 = env_graph.copy() # Agent's world model (doesn't know any edges are blocked)

        current_node = path_1[0]
        next_target_index = 1 # since 0 is the source
        index = 0
        total_travel_distance = 0.0

        while index < len(path_1) - 1:
            next_node = path_1[index + 1]

            # Update the next_target_index 
            if current_node == target_nodes[next_target_index]:
                next_target_index += 1
            
            # Check out all the observable edges
            observable_edges = set(blocked_env_graph.nodes[current_node]["visible_edges"]) # From the actual blocked env graph
            assumed_observable_edges = set(env_graph1.nodes[current_node]["visible_edges"]) # From the agent's world model 
            blocked_edges = assumed_observable_edges - observable_edges
            
            if len(blocked_edges) > 0:
                # Remove blocked edges from agent's world model
                for edge in blocked_edges:
                    u, v = edge
                    if env_graph1.has_edge(u, v):
                        # print(f"From {current_node}, observed edge ({u}, {v}) is blocked. Removing from graph.")
                        env_graph1.remove_edge(u, v)
                        # Update the visibility mapping as well
                        for node in env_graph1.nodes():
                            if "visible_edges" in env_graph1.nodes[node]:
                                current_visible = env_graph1.nodes[node]["visible_edges"]
                                # Remove any blocked edges from the visible_edges set
                                updated_visible = set(current_visible) - blocked_edges
                                env_graph1.nodes[node]["visible_edges"] = list(updated_visible)
                
                # Check if any blocked edge is in our current path
                path_edges = [(path_1[i], path_1[i+1]) for i in range(index, len(path_1) - 1)]

                # Create a set that includes both directions of blocked edges
                blocked_edges_both_directions = set()
                for u, v in blocked_edges:
                    blocked_edges_both_directions.add((u, v))
                    blocked_edges_both_directions.add((v, u))
                
                if any(edge in blocked_edges_both_directions for edge in path_edges):
                    # print(f"Blocked edge detected in planned path. Recalculating entire remaining path...")
                    
                    # Replan through ALL remaining targets
                    remaining_targets = target_nodes[next_target_index:]
                    new_path = [current_node]
                    
                    for target in remaining_targets:
                        segment = nx.shortest_path(env_graph1, source=new_path[-1], target=target, weight="distance")
                        new_path.extend(segment[1:])  # Append segment excluding the first node (already in new_path)
                    
                    # Replace the rest of path_1 with the new path
                    path_1 = path_1[:index + 1] + new_path[1:]

            # Get the next node from the (possibly updated) path
            next_node = path_1[index + 1]
            
            # Now that we have a traversable path, just go to the next node in path
            # print(f"Moving from {current_node} to {next_node}")
            total_travel_distance += env_graph.edges[current_node, next_node]["distance"]
            current_node = next_node
            index += 1

        path_length_list.append(total_travel_distance)
        
    end_time = time.time()
    runtimes.append(end_time - start_time)
        
if path_length_list:
    mean_cost = np.mean(path_length_list)
    variance_cost = np.var(path_length_list)
    std_dev = np.std(path_length_list)
    avg_runtime = np.mean(runtimes)

    print("\n" + "="*30)
    print(f"RESULTS FOR {len(path_length_list)} SUCCESSFUL RUNS")
    print("-" * 30)
    print(f"Mean Path Cost:     {mean_cost:.2f}")
    print(f"Variance:           {variance_cost:.2f}")
    print(f"Std Deviation:      {std_dev:.2f}")
    print(f"Avg Runtime/Run:    {avg_runtime:.4f}s")
    print("="*30)