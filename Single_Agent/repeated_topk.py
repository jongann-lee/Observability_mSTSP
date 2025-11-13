import numpy as np
import networkx as nx

from Single_Agent.reward_functions import visibility_reward
from Single_Agent.lin_kernighan_tsp import solve_tsp_lin_kernighan

def calculate_path_reward(path, env_graph: nx.Graph, reward_ratio: float) -> float:
    """
    Calculates the total reward for a given path in the enviroment graph.

    The environment graph contains nodes with a 'visible_edges' attribute,
    the edges have a 'distance', 'observed_edge', 'num_used' attribute. 

    They are used to calculate reward function, which is defined in reward_functions.py

    Note that we do not calculate the visibility reward at the final node in the path for consistency.

    Args:
        path (list): A list of nodes representing the path taken by the agent.
        env_graph (nx.Graph): The environment graph containing nodes and edges with attributes.
        reward_ratio (float): A weighting factor to balance visibility reward and distance penalty.
    Returns: 
        float: The total calculated reward for the given path.
    """

    total_visibility_reward = 0.0
    total_distance = 0.0

    for i in range(len(path) - 1):

        current_node = path[i]
        next_node = path[i+1]
        
        # Calculate visibility reward at the current node
        node_visibility_reward = visibility_reward(env_graph, current_node)
        total_visibility_reward += node_visibility_reward

        # Calculate the distance to the next node
        edge_distance = env_graph.edges[current_node, next_node]["distance"]
        total_distance += edge_distance

        # Update the visibility of edges after observing from the current node
        if "visible_edges" in env_graph.nodes[current_node]:
            visible_edges = env_graph.nodes[current_node]["visible_edges"]
            visible_unexplored_edges = [edge for edge in visible_edges if env_graph.edges[edge]["observed_edge"] == False]
        if len(visible_unexplored_edges) > 0:
            for edge in visible_unexplored_edges:
                env_graph.edges[edge]["observed_edge"] = True

        # Move to the next node in the path
        current_node = next_node
    
    # Calculate the final reward

    total_reward = reward_ratio * total_visibility_reward - total_distance

    return total_reward
    
        
class RepeatedTopK:
    """
    The Repeated TopK method for determining the best path for exploration and target reaching
    """
    def __init__(self, reward_ratio: float, env_graph: nx.Graph, target_graph: nx.Graph):
        """
        Initializes the RepeatedTopK method with the specified parameters.

        Args:
            k (int): The number of top paths to consider at each step.
            reward_ratio (float): A weighting factor to balance visibility reward and distance penalty.
        """
        self.reward_ratio = reward_ratio
        self.env_graph = env_graph
        self.target_graph = target_graph

    def generate_Hamiltonian_path(self):
        """
        Generates a Hamiltonian path using NetworkX's TSP approximation.
        Since TSP returns a cycle, we remove the last node to get a path.
        
        Args:
            num_path: Number of paths (currently only returns 1)
            target_graph: A fully connected graph with nodes to visit
            
        Returns:
            the Hamiltonian path
        """
        
        # Find the source node to use as starting point
        source_node = None
        for node in self.target_graph.nodes():
            if self.target_graph.nodes[node].get('type') == 'source':
                source_node = node
                break
        if source_node is None:
            raise ValueError("No source node found in the target graph.")
        
        # Get TSP cycle 
        node_list = list(self.target_graph.nodes())
        distance_matrix = nx.to_numpy_array(self.target_graph, nodelist=node_list, weight='distance', nonedge=np.inf)
        tsp_cycle, _ = solve_tsp_lin_kernighan(distance_matrix)
        tsp_cycle = [node_list[i] for i in tsp_cycle]
        print(f"TSP Cycle: {tsp_cycle}")
        
        if len(tsp_cycle) > 1 and tsp_cycle[0] == tsp_cycle [-1]:
            # If it's a cycle, remove the last node to make it a path
            tsp_path_nodes = tsp_cycle[:-1]
        else:
            # Fallback in case the cycle isn't returned as expected
            tsp_path_nodes = tsp_cycle

        # Rotate the cycle to start from it
        try:
            source_idx = tsp_path_nodes.index(source_node)
        except ValueError:
            raise ValueError(f"Source node {source_node} not found in TSP path {tsp_path_nodes}")
        hamiltonian_path = tsp_path_nodes[source_idx:] + tsp_path_nodes[:source_idx]
                
        return hamiltonian_path
    
    def deviate_path_at_node(self, base_path, node_in_path):
        """
        Creating an alternate to the base path by deviating at a certain node_in_path 
        """
        # Find the index of the deviation node in the base path
        if node_in_path not in base_path:
            return None
        
        deviation_idx = base_path.index(node_in_path)
        
        # Can't deviate at the end node
        if deviation_idx == len(base_path) - 1:
            return None
        
        # Get all neighbors of the deviation node
        neighbors = list(self.env_graph.neighbors(node_in_path))
        
        # Filter out neighbors that are already in the base path
        # (we want to deviate to a node NOT in the original path)
        valid_neighbors = [n for n in neighbors if n not in base_path]
        
        if not valid_neighbors:
            # No valid deviation possible
            return None
        
        # Randomly select a neighbor to deviate to
        deviation_target = np.random.choice(valid_neighbors)
        
        # Build the augmented path:
        # 1. Keep the path up to the deviation point
        augmented_path = base_path[:deviation_idx + 1]
        
        # 2. Add the deviation node
        augmented_path.append(deviation_target)
        
        # 3. Find shortest path from deviation_target back to the original destination
        destination = base_path[-1]
        try:
            return_path = nx.shortest_path(
                self.env_graph, 
                source=deviation_target, 
                target=destination, 
                weight="distance"
            )
            # Add the return path (excluding the first node since it's already added)
            augmented_path.extend(return_path[1:])

        except nx.NetworkXNoPath:
            # No path back to destination
            return None
        
        return augmented_path



    def process_section(self, begin_node, end_node):
        """
        Processes a single edge from the target graph and finds the best path
        """

        # Ensure that the begin_node and the end_node are actually neighbors in the target_graph

        if begin_node not in self.target_graph.nodes() or end_node not in self.target_graph.nodes():
            raise ValueError("Begin or end node not in target graph.")
        
        if not self.target_graph.has_edge(begin_node, end_node):
            raise ValueError("The specified edge does not exist in the target graph.")


        diverse_paths_data = self.target_graph.edges[begin_node, end_node]['diverse_paths']
        base_paths = [p['path'] for p in diverse_paths_data]

        best_reward = -np.inf
        best_path = None
        
        for path in base_paths:

            # Check if the path is in the REVERSE order
            if path[0] == end_node and path[-1] == begin_node:
                path.reverse() # Fix it: [7, ..., 50] -> [50, ..., 7]
            
            # Calculate the reward for the base path
            reward = calculate_path_reward(path, self.env_graph.copy(), self.reward_ratio)
            if reward > best_reward:
                best_reward = reward
                best_path = path

            # Now consider deviations at each node in the path (except the last)
            for node in path[:-1]:
                deviated_path = self.deviate_path_at_node(path, node)
                if deviated_path is not None:
                    reward = calculate_path_reward(deviated_path, self.env_graph.copy(), self.reward_ratio)
                    if reward > best_reward:
                        best_reward = reward
                        best_path = deviated_path

        # We don't return the best reward since it's not correct for the overall path
        return best_path
    

    def find_best_path(self):
        """
        Finds the nearly best Hamiltonian path using the given method
        """

        base_Hamiltonian_path = self.generate_Hamiltonian_path()

        best_path = []

        # Process each section of the Hamiltonian path
        for i in range(len(base_Hamiltonian_path) - 1):
            begin_node = base_Hamiltonian_path[i]
            end_node = base_Hamiltonian_path[i + 1]

            print(f"Processing section from {begin_node} to {end_node}")

            section_best_path = self.process_section(begin_node, end_node)

            # Append the section best path, avoiding duplication of nodes at the end
            if len(best_path) > 0 and best_path[-1] == section_best_path[0]:
                best_path.extend(section_best_path[1:])
            else:
                best_path.extend(section_best_path)

        # OPTIONAL: Calculate the total reward of the best path and compare to the original path
        best_path_reward = calculate_path_reward(best_path, self.env_graph.copy(), self.reward_ratio)
        
        original_path = []
        for i in range(len(base_Hamiltonian_path) - 1):
            begin_node = base_Hamiltonian_path[i]
            end_node = base_Hamiltonian_path[i + 1]
            edge_path = self.target_graph.edges[begin_node, end_node]['diverse_paths'][0]['path']
            if len(original_path) > 0 and original_path[-1] == edge_path[0]:
                original_path.extend(edge_path[1:])
            else:
                original_path.extend(edge_path)
        original_path_reward = calculate_path_reward(original_path, self.env_graph.copy(), self.reward_ratio)
        print(f"Original Hamiltonian Path Reward: {original_path_reward}")
        print(f"Best Path Reward: {best_path_reward}")

        return best_path
            




