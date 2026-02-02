import networkx as nx
import numpy as np

class VOITrajectoryManager:
    def __init__(self, target_nodes, chokepoints_list, block_prob=0.3):
        """
        Initializes the trajectory manager with a list of potential chokepoints.
        
        Args:
            env_graph (nx.Graph): The agent's initial world model.
            target_nodes (list): Sequence of Hamiltonian targets to visit.
            chokepoints_list (list): List of edge tuples (u, v) that could be blocked.
            block_prob (float): Prior probability of a chokepoint being blocked.
        """
    
        self.target_nodes = target_nodes
        self.block_prob = block_prob
        self.chokepoints_list = chokepoints_list 

    def calculate_path_cost(self, path, graph):
        """Calculates total distance of a path in a given graph."""
        return sum(graph.edges[u, v]['distance'] for u, v in zip(path[:-1], path[1:]))
    
    def find_least_deviating_obs_node(self, path_1, candidate_obs_nodes):
        """
        Finds the observation node that is closest to the original planned path.
        
        Args:
            path_1 (list): The list of nodes in the shortest path.
            candidate_obs_nodes (list): A list of potential observation node IDs.
            
        Returns:
            int/str: The observation node ID that is closest to path_1.
        """
        best_obs_node = None
        min_deviation_dist = float('inf')
        
        # Check distance from every candidate to every node in path_1
        for obs_node in candidate_obs_nodes:
            for path_node in path_1:
                try:
                    # Get distance from path node to observation node
                    dist = nx.shortest_path_length(self.world_model, path_node, obs_node, weight="distance")
                    
                    if dist < min_deviation_dist:
                        min_deviation_dist = dist
                        best_obs_node = obs_node
                except nx.NetworkXNoPath:
                    continue
                    
        return best_obs_node

    def get_voi_path(self, env_graph, current_node, target_node, obs_node_list):
        """
        Calculates if the agent should visit an observation node first.

        Does this by comparing the expected cost vs benefit of going to the visiblity node

        Path 1: Shortest path, not blocked

        Path 2: Shortest path, blocked (backtrack)

        Path 3: Visibility node, then shortest path (not blocked)

        Path 4: Visibility node, then shortest path (blocked)

   

        returns: the path to go down (assume chokepoint won't be blocked so, path 1 or path 3)
        """
        self.world_model = env_graph.copy()

        # --- PATH 1: Standard Shortest Path ---
        path_1 = nx.shortest_path(self.world_model, current_node, target_node, weight="distance")
        cost_1 = self.calculate_path_cost(path_1, self.world_model)

        # Identify which chokepoint from the list is on Path 1
        path_edges = set()
        for i in range(len(path_1) - 1):
            u, v = path_1[i], path_1[i+1]
            path_edges.add(tuple(sorted((u, v))))

        active_chokepoint = None
        for u_chk, v_chk in self.chokepoints_list:
            if tuple(sorted((u_chk, v_chk))) in path_edges:
                active_chokepoint = (u_chk, v_chk)
                break

        # If no chokepoint is on the current path, VOI logic is trivial (Path 1)
        if not active_chokepoint:
            return path_1

        u_chk, v_chk = active_chokepoint
        
        # --- EXPECTED COST: NO SENSING --- 
        # Generate world where the specific active chokepoint is blocked
        graph_blocked = self.world_model.copy()
        graph_blocked.remove_edge(u_chk, v_chk)
        
        # Path 2: Travel to chokepoint, discover it's blocked, then reroute (Backtracking cost) [cite: 98, 214]
        path_to_chk = nx.shortest_path(self.world_model, current_node, u_chk, weight="distance")
        path_from_chk = nx.shortest_path(graph_blocked, u_chk, target_node, weight="distance")
        path_2 = path_to_chk + path_from_chk[1:]
        cost_2 = self.calculate_path_cost(path_2, graph_blocked)

        e_no_sense = (1 - self.block_prob) * cost_1 + (self.block_prob * cost_2)

        # --- EXPECTED COST: SENSING --- 

        obs_node = self.find_least_deviating_obs_node(path_1, obs_node_list)
        if not obs_node:
            return path_1
        path_to_obs = nx.shortest_path(self.world_model, current_node, obs_node, weight="distance")
        
        # Path 3: Sensing node, then shortest path (if open)
        path_obs_open = nx.shortest_path(self.world_model, obs_node, target_node, weight="distance")
        path_3 = path_to_obs + path_obs_open[1:]
        cost_3 = self.calculate_path_cost(path_3, self.world_model)

        # Path 4: Sensing node, then shortest path (if blocked)
        path_obs_blocked = nx.shortest_path(graph_blocked, obs_node, target_node, weight="distance")
        path_4 = path_to_obs + path_obs_blocked[1:]
        cost_4 = self.calculate_path_cost(path_4, graph_blocked)

        e_sense = (1 - self.block_prob) * cost_3 + (self.block_prob * cost_4)

        # Compare costs to determine trajectory
        if e_no_sense <= e_sense:
            return path_1  
        else:
            return path_3