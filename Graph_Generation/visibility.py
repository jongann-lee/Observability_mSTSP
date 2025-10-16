import numpy as np
import networkx as nx


# This module defines the visibility mapping

def line_of_sight_visibility(input_graph: nx.Graph, visibility_range: int = None) -> nx.Graph:
    """
    Calculates the line-of-sight visibility graph from an input grid graph.

    An edge is considered visible to if they it is in the same row or column
    as the node and there is an unbroken path of  between them in the input graph.
    The visibility can be limited by a maximum range.

    Args:
        input_graph (nx.Graph): The input graph, where nodes are assumed to be 
                                (row, col) or (x, y) tuples.
        visibility_range (int, optional): The maximum number of grid cells a node 
                                          can see. If None, visibility is unlimited 
                                          across the grid's extent. Defaults to None.

    Returns:
        nx.Graph: A new graph containing edges between all mutually visible nodes.
    """
    # Use a set for efficient O(1) average time complexity for node lookups
    nodes_set = set(input_graph.nodes())
    if not nodes_set:
        # Set an empty attribute for an empty graph and return
        nx.set_node_attributes(input_graph, {}, name='visibility')
        return input_graph

    # Find chains of edges where any node inside that chain can see all of the edges in it
    horizontal_chains = []
    vertical_chains = []

    # Determine the grid's bounding box to define scan area
    min_x = min(n[0] for n in nodes_set)
    max_x = max(n[0] for n in nodes_set)
    min_y = min(n[1] for n in nodes_set)
    max_y = max(n[1] for n in nodes_set)

    """
    Finds all continuous horizontal and vertical chains by scanning the grid
    and checking for both node and edge existence at each step.
    """
    horizontal_chains = []
    vertical_chains = []
    
    # A set of nodes for fast lookups
    nodes_set = set(input_graph.nodes())
    # A canonical set of edges for fast lookups
    edge_set = {tuple(sorted(edge)) for edge in input_graph.edges()}

    # Get the grid's bounding box to define the scan area
    if not nodes_set:
        return [], []
    min_x = min(n[0] for n in nodes_set)
    max_x = max(n[0] for n in nodes_set)
    min_y = min(n[1] for n in nodes_set)
    max_y = max(n[1] for n in nodes_set)

    # --- Horizontal Scan ---
    for r in range(min_x, max_x + 1):
        current_chain = []
        for c in range(min_y, max_y + 1):
            node = (r, c)
            
            # If the node doesn't exist, any current chain is broken.
            if node not in nodes_set:
                if len(current_chain) > 1:
                    horizontal_chains.append(current_chain)
                current_chain = []
                continue

            # The node exists.
            if not current_chain:
                # If there's no active chain, start a new one with this node.
                current_chain.append(node)
            else:
                # If a chain is active, check for an edge from the previous node.
                previous_node = current_chain[-1]
                if tuple(sorted((previous_node, node))) in edge_set:
                    # If connected, add it to the chain.
                    current_chain.append(node)
                else:
                    # If not connected, the old chain ends...
                    if len(current_chain) > 1:
                        horizontal_chains.append(current_chain)
                    # ...and a new one begins with the current node.
                    current_chain = [node]
        
        # After the row scan is complete, save any remaining valid chain.
        if len(current_chain) > 1:
            horizontal_chains.append(current_chain)
            
    # --- Vertical Scan (Same logic, just swapping axes) ---
    for c in range(min_y, max_y + 1):
        current_chain = []
        for r in range(min_x, max_x + 1):
            node = (r, c)
            if node not in nodes_set:
                if len(current_chain) > 1:
                    vertical_chains.append(current_chain)
                current_chain = []
                continue
            
            if not current_chain:
                current_chain.append(node)
            else:
                previous_node = current_chain[-1]
                if tuple(sorted((previous_node, node))) in edge_set:
                    current_chain.append(node)
                else:
                    if len(current_chain) > 1:
                        vertical_chains.append(current_chain)
                    current_chain = [node]

        if len(current_chain) > 1:
            vertical_chains.append(current_chain)


    # Create a lookup map from each node to the chain(s) it belongs to
    node_to_chain = {node: {} for node in nodes_set}
    for chain in horizontal_chains:
        chain_tuple = tuple(chain) # Use tuple for immutability
        for node in chain:
            node_to_chain[node]['horizontal'] = chain_tuple
    for chain in vertical_chains:
        chain_tuple = tuple(chain)
        for node in chain:
            node_to_chain[node]['vertical'] = chain_tuple

    # For each node, define its visibility as the union of its chains
    visibility_mapping = {}
    for node in input_graph.nodes():
        visible_edges = set()
        
        # Helper to convert a list of nodes into a set of canonical edge tuples
        def nodes_to_edges(node_list: list) -> set:
            edges = set()
            for i in range(len(node_list) - 1):
                # Sort nodes within the tuple to create a canonical edge representation
                # e.g., ((1,0), (0,0)) becomes ((0,0), (1,0))
                edge = tuple(sorted((node_list[i], node_list[i+1])))
                edges.add(edge)
            return edges
        
        # --- Horizontal Visibility ---
        if 'horizontal' in node_to_chain.get(node, {}):
            h_chain = node_to_chain[node]['horizontal']
            if visibility_range is None:
                visible_edges.update(nodes_to_edges(h_chain))
            else:
                node_index = h_chain.index(node)
                start = max(0, node_index - visibility_range)
                end = min(len(h_chain), node_index + visibility_range + 1)
                # Convert the slice of visible nodes into edges
                visible_edges.update(nodes_to_edges(h_chain[start:end]))

        # --- Vertical Visibility ---
        if 'vertical' in node_to_chain.get(node, {}):
            v_chain = node_to_chain[node]['vertical']
            if visibility_range is None:
                visible_edges.update(nodes_to_edges(v_chain))
            else:
                node_index = v_chain.index(node)
                start = max(0, node_index - visibility_range)
                end = min(len(v_chain), node_index + visibility_range + 1)
                # Convert the slice of visible nodes into edges
                visible_edges.update(nodes_to_edges(v_chain[start:end]))
            
        visibility_mapping[node] = list(visible_edges)
    
    nx.set_node_attributes(input_graph, visibility_mapping, name="visible_edges")
    return input_graph