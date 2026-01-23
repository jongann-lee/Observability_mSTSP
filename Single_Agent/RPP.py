import numpy as np
import networkx as nx
import math

def calculate_entropy(probabilities):
    """Calculates Shannon entropy for a distribution of realizations."""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def get_conditional_entropy(Y, observation_node, realizations, env_graph):
    """
    Calculates H(X_Y | O). Since sensing is fixed at nodes, we check the 
    status of 'visible_edges' in each possible realization.
    """
    visible_edges = env_graph.nodes[observation_node].get("visible_edges", [])
    if not visible_edges:
        raise ValueError(f"Node {observation_node} has no visible edges and cannot be constructive.")

    # Group realizations by the 'outcome' (which edges are blocked/unblocked)
    outcomes = {}
    total_prob_Y = sum(realizations[i]['prob'] for i in Y)
    
    for i in Y:
        # Outcome is a tuple of (edge, status) for all visible edges
        outcome = tuple((e, e in realizations[i]['edges']) for e in visible_edges)
        outcomes.setdefault(outcome, []).append(i)

    cond_entropy = 0.0
    for outcome, indices in outcomes.items():
        prob_outcome = sum(realizations[i]['prob'] for i in Y) / total_prob_Y
        # Entropy of the subset of realizations consistent with this outcome
        subset_probs = [realizations[i]['prob'] for i in indices]
        norm_subset_probs = [p / sum(subset_probs) for p in subset_probs]
        cond_entropy += prob_outcome * calculate_entropy(norm_subset_probs)
        
    return cond_entropy

class ZeroCostRPP:
    def __init__(self, env_graph, realizations, start_node, goal_node):
        """
        Args:
            env_graph: nx.Graph with 'visible_edges' on nodes.
            realizations: List of dicts {'prob': float, 'edges': set_of_free_edges}.
            start_node: Start vertex s.
            goal_node: Goal vertex g.
        """
        self.env_graph = env_graph
        self.realizations = realizations
        self.s = start_node
        self.g = goal_node
        self.policy_tree = nx.DiGraph()

    def get_expected_cost_to_go(self, u, Y):
        """Calculates expected distance to goal across valid realizations[cite: 337]."""
        total_cost = 0.0
        total_prob = sum(self.realizations[i]['prob'] for i in Y)
        
        for i in Y:
            prob_i = self.realizations[i]['prob'] / total_prob
            # Create a temporary subgraph of only free edges in realization i
            free_edges = self.realizations[i]['edges']
            # Compute shortest path in realization i
            try:
                # We assume edges in realizations have 'distance' weights
                dist = nx.shortest_path_length(self.env_graph.edge_subgraph(free_edges), 
                                               u, self.g, weight="distance")
                total_cost += prob_i * dist
            except nx.NetworkXNoPath:
                continue # Cost is 0 for unreachable (per paper's terminal state logic) 
        
        return total_cost

    def plan(self):
        """Builds the policy tree (Algorithm 1)."""
        # Queue stores (consistent_indices_Y, current_node)
        # 1. Ensure Y is a tuple (hashable) so it can be a NetworkX node 
        initial_belief = (tuple(range(len(self.realizations))), self.s)
        Q = [initial_belief] 
        self.policy_tree.add_node(initial_belief)

        while Q:
            Y_tuple, v = Q.pop(0)
            Y = list(Y_tuple)
            
            # 1. Check if goal is unreachable in all possible realizations 
            if all(not nx.has_path(self.env_graph.edge_subgraph(self.realizations[i]['edges']), v, self.g) for i in Y):
                self.policy_tree.nodes[(tuple(Y), v)]['type'] = 'terminal_no_goal'
                continue

            # 2. Find reachable constructive observations 
            # In your case: nodes where sensing provides info and are reachable via 'known' edges.
            known_edges = [e for e in self.env_graph.edges if all(e in self.realizations[i]['edges'] for i in Y)]
            known_subgraph = self.env_graph.edge_subgraph(known_edges)
            
            reachable_nodes = nx.single_source_dijkstra_path_length(known_subgraph, v, weight="distance")
            
            best_node = None
            min_score = float('inf')

            for u, dist_to_u in reachable_nodes.items():

                # PAPER LOGIC: Only consider "Constructive" observations (Definition 7)
                visible_edges = self.env_graph.nodes[u].get("visible_edges", [])
                
                # A node is constructive only if it produces different outcomes 
                # for different realizations currently in Y.
                is_constructive = False
                first_realization_outcome = None

                for i in Y:
                    outcome = tuple(sorted([e for e in visible_edges if e in self.realizations[i]['edges']]))
                    if first_realization_outcome is None:
                        first_realization_outcome = outcome
                    elif outcome != first_realization_outcome:
                        is_constructive = True
                        break
                        
                if not is_constructive:
                    continue # Skip nodes that can't help narrow down Y

                # Pruning logic: If dist(v,g) <= dist(v,u) + expected_dist(u,g), skip u 
                try:
                    current_known_dist_to_g = nx.shortest_path_length(known_subgraph, v, self.g, weight="distance")
                    expected_dist_via_u = dist_to_u + self.get_expected_cost_to_go(u, Y)
                    if current_known_dist_to_g <= expected_dist_via_u:
                        continue 
                except nx.NetworkXNoPath:
                    pass # Keep searching for info if no known path to goal exists

                # 3. Calculate Score (Minimal Conditional Entropy + Tie-break with Distance)
                h_cond = get_conditional_entropy(Y, u, self.realizations, self.env_graph)
                # Equation 10 logic: score = Expected_Cost + H(X|O)
                score = (dist_to_u + self.get_expected_cost_to_go(u, Y)) + h_cond
                
                if score < min_score:
                    min_score = score
                    best_node = u

            # 4. Branching
            if best_node is None or best_node == self.g:
                # Move to goal [cite: 411]
                self.policy_tree.add_edge((tuple(Y), v), "GOAL", action="move_to_g")
                print("Added Goal node, we can get to the goal directly")
            else:
                # Add the chosen leg (from current v to the best observation node u)
                self.policy_tree.add_node((tuple(Y), best_node), type='observation_point')
                print("Added observation node:", best_node)
                self.policy_tree.add_edge((tuple(Y), v), (tuple(Y), best_node), action="move_to_obs")

                # Partition Y into outcomes based on the 'visible_edges' at best_node
                visible_edges = self.env_graph.nodes[best_node].get("visible_edges", [])
                outcomes = {}
                for i in Y:
                    # Sort using a consistent key (source, then target) to ensure outcomes match
                    outcome = tuple(sorted(
                        [e for e in visible_edges if e in self.realizations[i]['edges']],
                        key=lambda x: (x[0], x[1])
                    ))
                    outcomes.setdefault(outcome, []).append(i)

                # For each possible outcome, create a new belief state and add to queue
                for outcome, Y_E in outcomes.items():
                    new_state = (tuple(Y_E), best_node)
                    if new_state not in self.policy_tree:
                        self.policy_tree.add_node(new_state)
                        print("Added new belief state:", new_state)
                        # The edge represents the "result" of the sensing action
                        self.policy_tree.add_edge((tuple(Y), best_node), new_state, observation_result=outcome)
                        Q.append((Y_E, best_node)) # Continue planning from the new belief

        return self.policy_tree