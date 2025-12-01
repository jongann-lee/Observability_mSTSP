import numpy as np
import networkx as nx

from typing import List

def block_edge_on_path(env_graph: nx.Graph, path: List[int], block_per_section: int) -> nx.Graph:
    """
    Blocks edges on the given path in the environment graph.

    Parameters:
    - env_graph: The environment graph (networkx Graph).
    - path: A list of node indices representing the path.
    - block_per_section: Number of edges to block in each section of the path.
    
    Returns:
    - A new environment graph with specified edges blocked. 
        This is done with a boolean indicator. This edge may be removed later.
    """
    # Create a copy of the graph to avoid modifying the original
    blocked_graph = env_graph.copy()

    # Define a blocked attribute
    nx.set_edge_attributes(blocked_graph, False, 'blocked')
    
    # Identify target nodes (source and target_unreached) in the path
    target_indices = []
    for i, node in enumerate(path):
        node_type = env_graph.nodes[node].get("type", None)
        if node_type in ["source", "target_unreached"]:
            target_indices.append(i)
    
    # If there are no sections (less than 2 targets), return the graph as-is
    if len(target_indices) < 2:
        raise ValueError("Path must contain at least two target nodes to form sections.")
    
    # Process each section between consecutive target nodes
    for i in range(len(target_indices) - 1):
        start_idx = target_indices[i]
        end_idx = target_indices[i + 1]
        
        # Extract edges in this section
        section_edges = []
        for j in range(start_idx, end_idx):
            edge = (int(path[j]), int(path[j + 1]))
            section_edges.append(edge)
        
        # Randomly select edges to block
        if len(section_edges) > 0:
            num_to_block = min(block_per_section, len(section_edges))
            indices_to_block = np.random.choice(len(section_edges), num_to_block, replace=False)
            edges_to_block = [section_edges[int(idx)] for idx in indices_to_block]
            
            # Remove the selected edges from the graph
            for edge in edges_to_block:
                if blocked_graph.has_edge(edge[0], edge[1]):
                    blocked_graph.edges[edge[0], edge[1]]['blocked'] = True
    
    return blocked_graph    


def block_edges_maintain_connectivity(env_graph: nx.Graph, block_ratio: float) -> nx.Graph:
    """
    Blocks a specified number of edges randomly while ensuring the graph remains connected.
    
    Parameters:
    - env_graph: The environment graph (networkx Graph).
    - block_ratio: ratio of edges to block (set between 0 and 1)
    
    Returns:
    - A new environment graph with specified edges blocked (marked with 'blocked'=True).
      The graph connectivity is guaranteed to be maintained.
    """

    if block_ratio < 0 or block_ratio > 1:
        raise ValueError("block_ratio has to be between 0 and 1")

    # Create a copy of the graph to avoid modifying the original
    blocked_graph = env_graph.copy()

    num_edges_to_block = int(block_ratio * env_graph.number_of_edges())
    
    # Initialize blocked attribute if not present
    if not all('blocked' in blocked_graph.edges[e] for e in blocked_graph.edges()):
        nx.set_edge_attributes(blocked_graph, False, 'blocked')
    
    # Get all edges that are not already blocked
    available_edges = [e for e in blocked_graph.edges() if not blocked_graph.edges[e].get('blocked', False)]
    
    if len(available_edges) == 0:
        return blocked_graph
    
    # Shuffle edges for random selection
    np.random.shuffle(available_edges)
    
    blocked_count = 0
    attempted_edges = 0
    
    for edge in available_edges:
        if blocked_count >= num_edges_to_block:
            break
            
        attempted_edges += 1
        
        # Try blocking this edge
        blocked_graph.edges[edge[0], edge[1]]['blocked'] = True
        
        # Create a temporary graph with blocked edges removed to check connectivity
        temp_graph = blocked_graph.copy()
        edges_to_remove = [(u, v) for u, v, data in temp_graph.edges(data=True) if data.get('blocked', False)]
        temp_graph.remove_edges_from(edges_to_remove)
        
        # Check if graph is still connected
        if nx.is_connected(temp_graph):
            # Keep this edge blocked
            blocked_count += 1
        else:
            # Unblock this edge - it's critical for connectivity
            blocked_graph.edges[edge[0], edge[1]]['blocked'] = False
    
    return blocked_graph