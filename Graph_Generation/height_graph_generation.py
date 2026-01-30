import networkx as nx
import numpy as np
from collections import deque

class HeightMapGrid:
    def __init__(self, m, n):
        """
        Initializes a grid graph where every node has height 0.
        Source is (0,0), Target is (m-1, n-1).
        Distance weights are NOT calculated yet.
        """
        self.m = m
        self.n = n
        self.source = (0, 0)
        self.target = (m - 1, n - 1)
        
        # 1. Create the complete 2D grid
        self.G = nx.grid_2d_graph(m, n)
        
        # 2. Initialize "height" to 0.0
        nx.set_node_attributes(self.G, 0.0, name="height")

        # 3. Assign Types (source, target, intermediate)
        node_types = {node: "intermediate" for node in self.G.nodes()}
        node_types[self.source] = "source"
        node_types[self.target] = "target_unreached"
        nx.set_node_attributes(self.G, node_types, name="type")

        # 4. Assign 'pos' attribute
        pos_attributes = {node: node for node in self.G.nodes()}
        nx.set_node_attributes(self.G, pos_attributes, name="pos")

    def set_heights(self, nodes, height_value):
        """
        Assigns a specific height value to a list of nodes.
        
        Args:
            nodes (list of tuples): List of (r, c) coordinates.
            height_value (float): The height to assign to these nodes.
        """
        # Filter out invalid nodes just in case
        valid_nodes = [node for node in nodes if node in self.G]
        
        # Create a dict for the update
        height_update = {node: float(height_value) for node in valid_nodes}
        nx.set_node_attributes(self.G, height_update, name="height")

    def add_mountain(self, blob_nodes):
            """
            Takes a list of nodes (a 'blob') and assigns heights.
            Nodes in the blob directly connected to non-blob nodes get height 1.
            Nodes connected to those get height 2, and so on (BFS layering).
            """
            # Convert to set for O(1) lookup
            blob_set = set(node for node in blob_nodes if node in self.G)
            
            if not blob_set:
                return

            # 1. Find the "Boundary" (Height 1)
            # These are nodes in the blob that have at least one neighbor NOT in the blob
            queue = deque()
            visited_in_blob = set()
            new_heights = {}

            for node in blob_set:
                is_boundary = False
                for neighbor in self.G.neighbors(node):
                    if neighbor not in blob_set:
                        is_boundary = True
                        break
                
                if is_boundary:
                    # This node touches the "outside", so it starts at height 1
                    queue.append((node, 1))
                    visited_in_blob.add(node)
                    new_heights[node] = 1.0

            # 2. BFS to fill the center of the blob (Height 2, 3, ...)
            while queue:
                current_node, current_height = queue.popleft()
                
                for neighbor in self.G.neighbors(current_node):
                    # We only care about neighbors that are part of the mountain (blob)
                    # and haven't been assigned a height yet.
                    if neighbor in blob_set and neighbor not in visited_in_blob:
                        next_height = current_height + 1
                        visited_in_blob.add(neighbor)
                        new_heights[neighbor] = float(next_height)
                        queue.append((neighbor, next_height))
            
            # 3. Update the graph attributes
            # Nodes inside the blob that were "unreachable" (e.g. if the blob covers the WHOLE grid)
            # will simply retain their default height (0) or you can handle that edge case here.
            nx.set_node_attributes(self.G, new_heights, name="height")

    def calculate_distances(self):
        """
        Calculates edge weights based on:
        Base Distance (1.0) + 2 * |height_diff|
        """
        distances = {}
        
        for u, v in self.G.edges():
            # Get heights
            h_u = self.G.nodes[u]['height']
            h_v = self.G.nodes[v]['height']
            
            # Base grid distance is always 1.0 in a 4-connected grid
            # If you later allow diagonals, you might need euclidean distance here
            base_dist = 1.0 
            
            # Calculate cost
            weight = base_dist + 2 * abs(h_u - h_v)
            distances[(u, v)] = weight
            
        # Update the graph edges
        nx.set_edge_attributes(self.G, distances, name="distance")

        # Initialize the 'observed_edge' attirubtion to False
        nx.set_edge_attributes(self.G, False, name="observed_edge")


    def calculate_visibility(self):
            """
            Calculates recursive visibility based on height.
            1. You always see your immediate connected edges.
            2. If a neighbor is STRICTLY LOWER than you, you inherit everything they see.
            """
            # 1. Base Case: Initialize every node with its immediate incident edges
            visibility = {}
            for node in self.G.nodes():
                visible_edges = set()
                for neighbor in self.G.neighbors(node):
                    # Sort the tuple to ensure (A, B) and (B, A) are treated as the same edge
                    edge = tuple(sorted((node, neighbor)))
                    visible_edges.add(edge)
                visibility[node] = visible_edges

            # 2. Sort nodes by height (Ascending)
            # We process from the bottom of the valleys up to the peaks.
            # This ensures that when we process a node, all its lower neighbors 
            # have already finished calculating their full visibility.
            nodes_by_height = sorted(
                self.G.nodes(data=True), 
                key=lambda x: x[1]['height']
            )

            # 3. Propagate Visibility Uphill
            for node, data in nodes_by_height:
                current_height = data['height']
                
                for neighbor in self.G.neighbors(node):
                    neighbor_height = self.G.nodes[neighbor]['height']
                    
                    # The Core Logic: Inherit from lower neighbors
                    if neighbor_height < current_height:
                        visibility[node].update(visibility[neighbor])

            # 4. Save as node attribute (convert sets to lists)
            final_mapping = {k: list(v) for k, v in visibility.items()}
            nx.set_node_attributes(self.G, final_mapping, name="visible_edges")

    def get_graph(self):
        """Returns the internal NetworkX graph object."""
        return self.G