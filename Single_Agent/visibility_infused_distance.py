import networkx as nx
import numpy as np

def visibility_weighted_distance(env_graph: nx.graph,
                                 source_node,
                                 target_node,
                                 lambda_param: float) -> float:
    """
    The new distance measure is defined as 
    weighted = distance - lambda * visibility_reward(end node)

    Args:
        env_graph: NetworkX graph representing the environment
        source_node: Starting node of the edge
        target_node: Ending node of the edge
        lambda_param: Weighting parameter for vision reward (higher = more emphasis on vision)

    Returns:
        Modified edge weight (can be negative if vision reward is high enough)    
    """

    # Get base distance/weight of the edge
    if env_graph.has_edge(source_node, target_node):
        base_distance = env_graph.edges[source_node, target_node].get("distance")
    else:
        raise ValueError(f"No edge exists between {source_node} and {target_node}")

    visible_edges = env_graph.nodes[target_node]["visible_edges"]
    visible_unexplored_edges = [edge for edge in visible_edges if env_graph.edges[edge]["observed_edge"] == False]
    edge_value = np.array([env_graph.edges[edge]["num_used"] for edge in visible_unexplored_edges])
    visibility_reward = np.sum(edge_value)

    modified_weight = base_distance - lambda_param * visibility_reward

    return modified_weight



def add_weighted_distance(env_graph: nx.graph,
                        lambda_param: float) -> nx.digraph:
    """
    Creates a directed graph with an additional "visibility weighted distance" metric

    This metric is used to calculate the shortest distance whilest taking visibility into account

    The new distance measure is defined as 
    weighted = distance - lambda * visibility_reward(end node)

    Args:
        env_graph: Original NetworkX graph
        lambda_param: Weighting parameter for vision reward
        unreached_targets: List of targets that haven't been reached yet
    
    Returns:
        Directed graph with modified edge weights
    """

    # Create a directed graph
    modified_graph = nx.DiGraph()
    
    # Add all nodes from original graph
    modified_graph.add_nodes_from(env_graph.nodes(data=True))

    minimum_weight = float('inf')
    edge_weights = []
    
    # First pass: calculate all weights and find minimum
    for u, v, data in env_graph.edges(data=True):
        # Forward direction: u -> v
        forward_weight = visibility_weighted_distance(
            env_graph, u, v, lambda_param
        )
        edge_weights.append((u, v, forward_weight))
        minimum_weight = min(minimum_weight, forward_weight)
        
        # Backward direction: v -> u
        backward_weight = visibility_weighted_distance(
            env_graph, v, u, lambda_param
        )
        edge_weights.append((v, u, backward_weight))
        minimum_weight = min(minimum_weight, backward_weight)
    
    # Second pass: normalize weights by shifting to make all non-negative
    offset = abs(minimum_weight) if minimum_weight < 0 else 0
    
    for u, v, weight in edge_weights:
        normalized_weight = weight + offset
        modified_graph.add_edge(u, v, modified_distance=normalized_weight)
    
    return modified_graph