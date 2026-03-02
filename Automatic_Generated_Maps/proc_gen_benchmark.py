"""
Benchmark runner for procedurally generated maps.

Generates maps, benchmarks the RepeatedTopK agent and Shortest Path agent on
each map across multiple random blockage realizations, saves per-map results
to CSV, and pickles the top 3 / bottom 3 maps by improvement percentage.

Outputs:
  - benchmark_results.csv: Per-map benchmark results
  - benchmark_summary.json: Aggregate summary statistics
  - top_bottom_maps.pkl: Top 3 and bottom 3 maps by improvement percentage
"""

import sys
import os
import time
import json
import csv
import pickle
import argparse
import random

import numpy as np
import networkx as nx
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Graph_Generation.target_graph import create_fully_connected_target_graph
from Graph_Generation.edge_block_generation import block_specific_edges
from Single_Agent.repeated_topk import RepeatedTopK

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from map_generator import generate_map_suite


def run_shortest_path_agent(env_graph, blocked_env_graph, hamiltonian_target_path, source, target):
    """
    Runs the shortest-path replanning agent. Returns total travel distance.
    """
    shortest_path = []
    for i in range(len(hamiltonian_target_path) - 1):
        begin_node = hamiltonian_target_path[i]
        end_node = hamiltonian_target_path[i + 1]
        section_path = nx.shortest_path(env_graph, source=begin_node, target=end_node, weight="distance")
        if len(shortest_path) > 0 and shortest_path[-1] == section_path[0]:
            shortest_path.extend(section_path[1:])
        else:
            shortest_path.extend(section_path)

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

        observable_edges = set(blocked_env_graph.nodes[current_node].get("visible_edges", []))
        assumed_observable_edges = set(env_graph1.nodes[current_node].get("visible_edges", []))
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
                for t in remaining_targets:
                    try:
                        segment = nx.shortest_path(env_graph1, source=new_path[-1], target=t, weight="distance")
                        new_path.extend(segment[1:])
                    except nx.NetworkXNoPath:
                        return None  # Unreachable
                path_1 = path_1[:index + 1] + new_path[1:]

        next_node = path_1[index + 1]
        if not env_graph.has_edge(current_node, next_node):
            return None  # Edge doesn't exist
        total_travel_distance += env_graph.edges[current_node, next_node]["distance"]
        current_node = next_node
        index += 1

    return total_travel_distance


def run_our_agent(env_graph, blocked_env_graph, target_graph, hamiltonian_target_path,
                  source, target, reward_ratio=3.0, sample_recursion=4,
                  sample_num_obstacle=4, sample_obstacle_hop=1):
    """
    Runs the RepeatedTopK agent. Returns total travel distance.
    """
    env_graph2 = env_graph.copy()
    path2_generator = RepeatedTopK(
        reward_ratio=reward_ratio,
        env_graph=env_graph2,
        target_graph=target_graph,
        sample_recursion=sample_recursion,
        sample_num_obstacle=sample_num_obstacle,
        sample_obstacle_hop=sample_obstacle_hop
    )

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

        observable_edges = set(blocked_env_graph.nodes[current_node].get("visible_edges", []))
        assumed_observable_edges = set(env_graph2.nodes[current_node].get("visible_edges", []))
        blocked_edges = assumed_observable_edges - observable_edges

        # Mark observed edges
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
                for t in remaining_targets:
                    try:
                        segment = path2_generator.alternate_path_online(new_path[-1], t)
                        if segment is None:
                            return None
                        new_path.extend(segment[1:])
                    except Exception:
                        return None
                path_2 = path_2[:index + 1] + new_path[1:]

        next_node = path_2[index + 1]
        if not env_graph.has_edge(current_node, next_node):
            return None
        total_travel_distance += env_graph.edges[current_node, next_node]["distance"]
        current_node = next_node
        index += 1

    return total_travel_distance


def benchmark_single_map(map_data, num_runs=200, target_recursion=4,
                          target_num_obstacles=4, target_obstacle_hop=1):
    """
    Benchmarks both agents on a single generated map.

    Returns dict with per-run results for both agents.
    """
    env_graph = map_data['env_graph']
    chokepoints = map_data['chokepoints']
    source = map_data['source']
    target = map_data['target']
    block_prob = map_data['block_prob']

    # Create target graph
    try:
        target_graph = create_fully_connected_target_graph(
            env_graph,
            recursions=target_recursion,
            num_obstacles=target_num_obstacles,
            obstacle_hop=target_obstacle_hop
        )
    except Exception as e:
        return {'error': str(e), 'map_id': map_data.get('map_id', -1)}

    # Pre-compute Hamiltonian path
    try:
        path_generator = RepeatedTopK(
            reward_ratio=1.0, env_graph=env_graph.copy(), target_graph=target_graph,
            sample_recursion=4, sample_num_obstacle=4, sample_obstacle_hop=1
        )
        hamiltonian_target_path = path_generator.generate_Hamiltonian_path()
    except Exception as e:
        return {'error': str(e), 'map_id': map_data.get('map_id', -1)}

    sp_costs = []
    our_costs = []
    sp_runtimes = []
    our_runtimes = []
    valid_runs = 0

    for run_idx in range(num_runs):
        np.random.seed(42 + run_idx)
        random.seed(42 + run_idx)

        # Randomly block chokepoints
        edges_to_remove = []
        if len(chokepoints) > 0:
            RNG = np.random.rand(len(chokepoints))
            for i, edge in enumerate(chokepoints):
                if RNG[i] < block_prob:
                    u, v = edge
                    if env_graph.has_edge(u, v):
                        edges_to_remove.append((u, v))
                    elif env_graph.has_edge(v, u):
                        edges_to_remove.append((v, u))

        # Create blocked environment
        blocked_env_graph = block_specific_edges(env_graph, edges_to_remove)
        blocked_env_graph.remove_edges_from(edges_to_remove)

        # Skip if goal unreachable
        if not nx.has_path(blocked_env_graph, source, target):
            continue

        # Update visible_edges in blocked graph
        for node in blocked_env_graph.nodes():
            if "visible_edges" in blocked_env_graph.nodes[node]:
                current_visible = blocked_env_graph.nodes[node]["visible_edges"]
                edges_to_remove_set = set()
                for e in edges_to_remove:
                    edges_to_remove_set.add(tuple(sorted(e)))
                updated_visible = [e for e in current_visible
                                  if tuple(sorted(e)) not in edges_to_remove_set]
                blocked_env_graph.nodes[node]["visible_edges"] = updated_visible

        # --- Shortest Path Agent ---
        t0 = time.time()
        sp_cost = run_shortest_path_agent(
            env_graph.copy(), blocked_env_graph, hamiltonian_target_path, source, target
        )
        sp_time = time.time() - t0

        # --- Our Agent ---
        t0 = time.time()
        our_cost = run_our_agent(
            env_graph.copy(), blocked_env_graph, target_graph, hamiltonian_target_path,
            source, target
        )
        our_time = time.time() - t0

        if sp_cost is not None and our_cost is not None:
            sp_costs.append(sp_cost)
            our_costs.append(our_cost)
            sp_runtimes.append(sp_time)
            our_runtimes.append(our_time)
            valid_runs += 1

    return {
        'map_id': map_data.get('map_id', -1),
        'label': map_data.get('label', 'unknown'),
        'seed': map_data.get('seed', -1),
        'grid_size': map_data.get('grid_size', -1),
        'block_prob': block_prob,
        'num_blobs': len(map_data.get('blobs', [])),
        'num_chokepoints': len(chokepoints),
        'valid_runs': valid_runs,
        'sp_costs': sp_costs,
        'our_costs': our_costs,
        'sp_runtimes': sp_runtimes,
        'our_runtimes': our_runtimes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark agents on procedurally generated maps"
    )
    parser.add_argument("--num-maps", type=int, default=50,
                        help="Number of maps to generate and benchmark")
    parser.add_argument("--num-runs", type=int, default=200,
                        help="Number of blockage realizations per map")
    parser.add_argument("--seed-start", type=int, default=3000,
                        help="Starting seed for map generation")
    parser.add_argument("--output", type=str, default="benchmark_results.csv",
                        help="Output CSV filename")
    parser.add_argument("--output-summary", type=str, default="benchmark_summary.json",
                        help="Output summary JSON filename")
    parser.add_argument("--output-maps", type=str, default="top_bottom_maps.pkl",
                        help="Output pkl for top/bottom maps")
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(output_dir, args.output)
    output_json = os.path.join(output_dir, args.output_summary)
    output_pkl = os.path.join(output_dir, args.output_maps)

    # Generate maps
    print(f"\nGenerating {args.num_maps} maps (seed_start={args.seed_start})...")
    maps = generate_map_suite(num_maps=args.num_maps, seed_start=args.seed_start)
    print(f"Generated {len(maps)} structurally valid maps")

    # Benchmark all maps
    all_results = []  # (map_data, improvement_pct, result_dict)

    csv_fields = [
        'map_id', 'label', 'seed', 'grid_size', 'block_prob', 'num_blobs',
        'num_chokepoints', 'valid_runs',
        'sp_mean', 'sp_std', 'our_mean', 'our_std',
        'improvement_pct', 'our_wins_pct',
        'sp_avg_runtime', 'our_avg_runtime'
    ]

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

        for i, map_data in enumerate(tqdm(maps, desc="Benchmarking maps")):
            label = map_data.get('label', '?')
            seed = map_data.get('seed', '?')

            result = benchmark_single_map(map_data, num_runs=args.num_runs)

            if 'error' in result:
                print(f"  Map {i} ({label}, seed={seed}): ERROR: {result['error']}")
                continue

            if result['valid_runs'] == 0:
                print(f"  Map {i} ({label}, seed={seed}): SKIPPED (no valid runs)")
                continue

            sp_mean = np.mean(result['sp_costs'])
            sp_std = np.std(result['sp_costs'])
            our_mean = np.mean(result['our_costs'])
            our_std = np.std(result['our_costs'])
            improvement_pct = ((sp_mean - our_mean) / sp_mean) * 100 if sp_mean > 0 else 0.0

            wins = sum(1 for s, o in zip(result['sp_costs'], result['our_costs']) if o < s)
            win_pct = (wins / result['valid_runs']) * 100

            all_results.append((map_data, improvement_pct, result))

            # Write CSV row
            row = {
                'map_id': result['map_id'],
                'label': result['label'],
                'seed': result['seed'],
                'grid_size': result['grid_size'],
                'block_prob': result['block_prob'],
                'num_blobs': result['num_blobs'],
                'num_chokepoints': result['num_chokepoints'],
                'valid_runs': result['valid_runs'],
                'sp_mean': f"{sp_mean:.2f}",
                'sp_std': f"{sp_std:.2f}",
                'our_mean': f"{our_mean:.2f}",
                'our_std': f"{our_std:.2f}",
                'improvement_pct': f"{improvement_pct:.2f}",
                'our_wins_pct': f"{win_pct:.1f}",
                'sp_avg_runtime': f"{np.mean(result['sp_runtimes']):.4f}",
                'our_avg_runtime': f"{np.mean(result['our_runtimes']):.4f}",
            }
            writer.writerow(row)
            csvfile.flush()

            print(f"  Map {i:2d} ({label:20s}, seed={seed:5d}): "
                  f"improvement={improvement_pct:+6.2f}%, "
                  f"win_rate={win_pct:4.1f}%, "
                  f"SP={sp_mean:.1f}, Ours={our_mean:.1f}")

    # Sort by improvement
    all_results.sort(key=lambda x: -x[1])  # Best first

    print("\n" + "=" * 60)
    print(f"Completed {len(all_results)} maps")
    print("=" * 60)

    if len(all_results) == 0:
        print("No valid results!")
        return

    improvements = [r[1] for r in all_results]
    print(f"Mean improvement: {np.mean(improvements):.2f}%")
    print(f"Median improvement: {np.median(improvements):.2f}%")
    print(f"Std improvement: {np.std(improvements):.2f}%")
    print(f"Maps with positive improvement: {sum(1 for x in improvements if x > 0)}/{len(all_results)}")

    # Top 3 best and worst
    top3 = all_results[:3]
    bottom3 = all_results[-3:]

    print("\nTop 3 BEST maps:")
    for i, (md, imp, _) in enumerate(top3):
        print(f"  {i+1}. seed={md.get('seed')}, label={md.get('label')}, "
              f"improvement={imp:+.2f}%")

    print("\nTop 3 WORST maps:")
    for i, (md, imp, _) in enumerate(bottom3):
        print(f"  {i+1}. seed={md.get('seed')}, label={md.get('label')}, "
              f"improvement={imp:+.2f}%")

    # Save top/bottom maps pkl
    save_data = {
        'top3': [(md, imp) for md, imp, _ in top3],
        'bottom3': [(md, imp) for md, imp, _ in bottom3],
        'all_improvements': improvements,
    }
    with open(output_pkl, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved top/bottom maps to: {output_pkl}")

    # Aggregate summary JSON
    all_sp = []
    all_ours = []
    for _, _, r in all_results:
        all_sp.extend(r['sp_costs'])
        all_ours.extend(r['our_costs'])

    total_wins = sum(1 for s, o in zip(all_sp, all_ours) if o < s)
    total_ties = sum(1 for s, o in zip(all_sp, all_ours) if o == s)
    total_losses = sum(1 for s, o in zip(all_sp, all_ours) if o > s)
    total_runs = len(all_sp)

    summary = {
        'total_maps': len(all_results),
        'total_runs': total_runs,
        'overall_sp_mean': float(np.mean(all_sp)),
        'overall_our_mean': float(np.mean(all_ours)),
        'overall_improvement_pct': float(((np.mean(all_sp) - np.mean(all_ours)) / np.mean(all_sp)) * 100),
        'total_wins': total_wins,
        'total_ties': total_ties,
        'total_losses': total_losses,
        'win_rate_pct': float(total_wins / total_runs * 100) if total_runs > 0 else 0,
        'maps_with_positive_improvement': sum(1 for x in improvements if x > 0),
        'mean_per_map_improvement_pct': float(np.mean(improvements)),
        'median_per_map_improvement_pct': float(np.median(improvements)),
        'std_per_map_improvement_pct': float(np.std(improvements)),
    }

    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_csv}")
    print(f"Summary saved to: {output_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
