import numpy as np
import networkx as nx
import itertools

from .kspd import approximate_KSPD


### These are all the functions for creating a target only graph and calculating the shortest paths

def single_shortest_path(input_graph: nx.graph, output_graph: nx.graph, target_nodes):
    """
    Only compute a single shortest path
    """
    for u, v in itertools.combinations(target_nodes, 2):
        try:
            # 1. Find the shortest path (list of nodes) ONLY ONCE
            path_nodes = nx.shortest_path(input_graph, source=u, target=v, weight="distance")

            # 2. Derive the path's edges from the list of nodes
            # We convert the zip iterator to a list to reuse it
            path_edges = list(zip(path_nodes, path_nodes[1:]))

            # 3. Calculate the total path length from the path itself
            # This avoids a second call to a shortest path algorithm
            path_length = sum(input_graph.edges[edge]['distance'] for edge in path_edges)

            # Add the edge to the new target graph
            output_graph.add_edge(u, v, distance=path_length)

            # 4. Increment the 'num_used' counter for each edge in the path
            for edge in path_edges:
                input_graph.edges[edge]['num_used'] += 1

        except nx.NetworkXNoPath:
            # This case will only occur if the input_graph is not fully connected.
            # It's good practice to handle it.
            print(f"Warning: No path exists between {u} and {v} in the input graph.")

    return output_graph

def all_shortest_paths(input_graph: nx.Graph, output_graph: nx.Graph, target_nodes):
    """
    Compute all the shortest paths
    Add 1/n to the num_visited, with n being the number of shortest paths
    for each pair of target nodes.
    """
    # Iterate over every unique pair of target nodes
    for u, v in itertools.combinations(target_nodes, 2):
        try:
            # 1. Find all shortest paths. This returns a generator.
            all_paths_generator = nx.all_shortest_paths(
                input_graph, source=u, target=v, weight="distance"
            )

            # Convert generator to a list to count and reuse the paths
            all_paths = list(all_paths_generator)
            num_paths = len(all_paths)

            if num_paths > 0:
                # 2. Determine the credit to add for each path
                credit = 1.0 / num_paths

                # Calculate path length from the first path (all are the same length)
                first_path = all_paths[0]
                path_length = sum(
                    input_graph.edges[edge]['distance']
                    for edge in zip(first_path, first_path[1:])
                )
                output_graph.add_edge(u, v, distance=path_length)

                # 3. Distribute the credit across all edges in all paths
                for path in all_paths:
                    path_edges = zip(path, path[1:])
                    for edge in path_edges:
                        input_graph.edges[edge]['num_used'] += credit

        except nx.NetworkXNoPath:
            # This case will only occur if the input_graph is not fully connected.
            # It's good practice to handle it.
            print(f"Warning: No path exists between {u} and {v} in the input graph.")

    return output_graph

def top3_shortest_paths(input_graph: nx.Graph, output_graph: nx.Graph, target_nodes):
    """
    Compute the top 3 shortest paths (if they exist)
    The value is given in the following manner
    l1 < l2 < l3: weights are 4/7, 2/7, 1/7
    l1 < l2 = l3: weights are 1/2, 1/4, 1/4
    l1 = l2 < l3: weights are 2/5, 2/5, 1/5
    l1 = l2 = l3: weights are 1/3, 1/3, 1/3

    If there are only two paths
    l1 < l2: weights are 2/3, 1/3
    l1 = l2: weights are 1/2, 1/2

    If there's only one then obviously it's 1
    """
    k = 3 # Number of top paths to consider

    # Iterate over every unique pair of target nodes
    for u, v in itertools.combinations(target_nodes, 2):
        paths_generator = nx.shortest_simple_paths(input_graph, source=u, target=v, weight="distance")

        # Get the top k paths and calculate their lengths
        top_k_paths = []
        for path in itertools.islice(paths_generator, k):
            length = nx.path_weight(input_graph, path, weight="distance")
            top_k_paths.append({'path': path, 'length': length})

        # Make sure the shortest path exits
        if not top_k_paths:
            continue

        # Add the absolute shortest path info to the target graph
        shortest = top_k_paths[0]
        output_graph.add_edge(u, v, distance=shortest['length'])
    
        # WARNING: the weights are made with k=3 in mind

        if len(top_k_paths) == 1:
            # Only one path
            path_edges = zip(top_k_paths[0]['path'], top_k_paths[0]['path'][1:])
            for edge in path_edges:
                input_graph.edges[edge]['num_used'] += 1.0
        
        elif len(top_k_paths) == 2:
            # Two paths
            length1 = top_k_paths[0]['length']
            length2 = top_k_paths[1]['length']

            if length1 < length2:
                weights = [2/3, 1/3]
            elif length1 == length2:
                weights = [1/2, 1/2]

            for edge in zip(top_k_paths[0]['path'], top_k_paths[0]['path'][1:]):
                input_graph.edges[edge]['num_used'] += weights[0]
            for edge in zip(top_k_paths[1]['path'], top_k_paths[1]['path'][1:]):
                input_graph.edges[edge]['num_used'] += weights[1]

        elif len(top_k_paths) == 3:
            # Three paths
            length1 = top_k_paths[0]['length']
            length2 = top_k_paths[1]['length']
            length3 = top_k_paths[2]['length']

            if length1 < length2 < length3:
                weights = [4/7, 2/7, 1/7]
            elif length1 < length2 == length3:
                weights = [1/2, 1/4, 1/4]
            elif length1 == length2 < length3:
                weights = [2/5, 2/5, 1/5]
            elif length1 == length2 == length3:
                weights = [1/3, 1/3, 1/3]

            for i in range(3):
                for edge in zip(top_k_paths[i]['path'], top_k_paths[i]['path'][1:]):
                    input_graph.edges[edge]['num_used'] += weights[i]

        else: 
            # More than three paths (shouldn't happen with k=3)
            print(f"Warning: More than {k} paths found between {u} and {v}. This should not happen.")


    return output_graph


def top3_shortest_top4_closest(input_graph: nx.Graph, output_graph: nx.Graph, target_nodes):
    """
    Same as the top 3 version, only this time only the top 4 closest target nodes are connected

    Compute the top 3 shortest paths (if they exist)
    The value is given in the following manner
    l1 < l2 < l3: weights are 4/7, 2/7, 1/7
    l1 < l2 = l3: weights are 1/2, 1/4, 1/4
    l1 = l2 < l3: weights are 2/5, 2/5, 1/5
    l1 = l2 = l3: weights are 1/3, 1/3, 1/3

    If there are only two paths
    l1 < l2: weights are 2/3, 1/3
    l1 = l2: weights are 1/2, 1/2

    If there's only one then obviously it's 1
    """
    k = 3 # Number of top paths to consider
    num_neighbors = 4 # Number of closest neighbors to connect
    processed_pairs = set() # A set to keep track of pairs we've already analyzed

    # 1. Iterate through each target node as a starting point
    for u in target_nodes:
        neighbors = []
        for v in target_nodes:
            if u == v:
                continue
            distance = np.linalg.norm(np.array(u) - np.array(v))
            neighbors.append((distance, v))
            # ---------------------------------

        neighbors.sort(key=lambda x: x[0])
        closest_neighbors = neighbors[:num_neighbors]

        for _, v in closest_neighbors:
            pair = frozenset({u, v})
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            paths_generator = nx.shortest_simple_paths(input_graph, source=u, target=v, weight="distance")
            top_k_paths = [{'path': p, 'length': nx.path_weight(input_graph, p, 'distance')}
                           for p in itertools.islice(paths_generator, k)]

            if not top_k_paths:
                continue

            output_graph.add_edge(u, v, distance=top_k_paths[0]['length'])

            weights = []
            if len(top_k_paths) == 1:
                weights = [1.0]
            elif len(top_k_paths) == 2:
                l1, l2 = top_k_paths[0]['length'], top_k_paths[1]['length']
                weights = [2/3, 1/3] if l1 < l2 else [1/2, 1/2]
            elif len(top_k_paths) == 3:
                l1, l2, l3 = top_k_paths[0]['length'], top_k_paths[1]['length'], top_k_paths[2]['length']
                if l1 < l2 < l3: weights = [4/7, 2/7, 1/7]
                elif l1 < l2 == l3: weights = [1/2, 1/4, 1/4]
                elif l1 == l2 < l3: weights = [2/5, 2/5, 1/5]
                else: weights = [1/3, 1/3, 1/3]

            if weights:
                for i, p_data in enumerate(top_k_paths):
                    path_edges = zip(p_data['path'], p_data['path'][1:])
                    for edge in path_edges:
                        input_graph.edges[edge]['num_used'] += weights[i]


    return output_graph

def k_shortest_diverse_paths(input_graph: nx.Graph, output_graph: nx.Graph, target_nodes):
    """
    Creates a locally connected target graph with k nearest neighbors
    then find the top k shortest diverse paths between them
    """

    num_neighbors = 4
    k = 3
    tau = 0.25
    processed_pairs = set()

    # Get positions for all nodes
    pos = nx.get_node_attributes(input_graph, 'pos')

    # 1. Iterate through each target node as a starting point
    for u in target_nodes:
        neighbors = []
        for v in target_nodes:
            if u == v:
                continue
            # Get positions and calculate distance
            pos_u = np.array(pos[u])
            pos_v = np.array(pos[v])
            distance = np.linalg.norm(pos_u - pos_v)
            neighbors.append((distance, v))

        neighbors.sort(key=lambda x: x[0])
        closest_neighbors = neighbors[:num_neighbors]

        for _, v in closest_neighbors:
            pair = frozenset({u, v})
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            top_k_diverse_paths = approximate_KSPD(input_graph, source=u, target=v, k=k, tau=tau)

            if not top_k_diverse_paths:
                continue

            output_graph.add_edge(u, v, distance=top_k_diverse_paths[0]['length'])

            weights = []
            num_paths = len(top_k_diverse_paths)
            exponents = np.arange(num_paths - 1, -1, -1) 
            weights_numerator = np.power(2, exponents)   
            weights = weights_numerator / (2**num_paths - 1) 

            for i, p_data in enumerate(top_k_diverse_paths):
                path_edges = zip(p_data['path'], p_data['path'][1:])
                for edge in path_edges:
                    input_graph.edges[edge]['num_used'] += weights[i]
                # Store contributions for endpoint u
                if "stored_path_contributions" not in input_graph.nodes[u]:
                    input_graph.nodes[u]["stored_path_contributions"] = []
                for edge in path_edges:
                    input_graph.nodes[u]["stored_path_contributions"].append((edge, weights[i]))
                # Store contributions for endpoint v
                if "stored_path_contributions" not in input_graph.nodes[v]:
                    input_graph.nodes[v]["stored_path_contributions"] = []
                for edge in path_edges:
                    input_graph.nodes[v]["stored_path_contributions"].append((edge, weights[i]))


    return output_graph


def create_fully_connected_target_graph(input_graph: nx.Graph) -> nx.Graph:
    """
    Takes a graph with target nodes and creates a new, fully connected graph
    containing only those target nodes. The edge weights in the new graph
    are the shortest path distances between the nodes in the original graph.

    Args:
        input_graph (nx.Graph): The graph with nodes marked by a 'type'
                                attribute (e.g., 'target', 'source'). It
                                must have a 'distance' attribute on its edges.

    Returns:
        nx.Graph: A new, fully connected graph of just the target nodes.
    """
    # Initialize a 'num_used' counter on each edge of the input graph
    nx.set_edge_attributes(input_graph, 0, 'num_used')

    # Extract a list of nodes where the 'type' attribute is 'target'
    target_nodes = [
        node for node, data in input_graph.nodes(data=True)
        if data.get("type") == "target_unreached" or data.get("type") == "source"
    ]

    # Create a new, empty graph to store the result
    target_connected_graph = nx.Graph()
    
    # Copy nodes with their attributes, mapping 'target_unreached' -> 'target'
    for node in target_nodes:
        node_attrs = input_graph.nodes[node].copy()
        if node_attrs.get('type') == 'target_unreached':
            node_attrs['type'] = 'target'
        target_connected_graph.add_node(node, **node_attrs)

    # target_connected_graph = top3_shortest_top4_closest(input_graph, target_connected_graph, target_nodes)
    target_connected_graph = k_shortest_diverse_paths(input_graph, target_connected_graph, target_nodes)

    return target_connected_graph