import numpy as np
import networkx as nx

from typing import Dict, Tuple, List, Set, Optional
from .reward_functions import target_and_visibility_reward

class SuboptimalValueIteration_Simple:
    """
    Implementation of Suboptimal Value Iteration for multi-agent Steiner TSP
    with uncertain edges and visibility constraints.
    NOTE: This is the version that only works with edges of length 1
    """
    
    def __init__(self, env_graph: nx.Graph, s0, epsilon: float = 0.01, gamma: float = 0.1):
        """
        Initialize the value iteration algorithm.
        
        Args:
            env_graph: NetworkX graph representing the environment
            s0: Initial state 
            epsilon: Convergence threshold for value iteration
            gamma: Discount factor
        """
        self.env_graph = env_graph
        self.s0 = s0
        self.epsilon = epsilon
        self.gamma = gamma

        self.unreached_targets = [node for node, data in self.env_graph.nodes(data=True) 
                                 if data.get("type") == "target_unreached"]
        
        # Initialize value functions
        self.V_t = {}  # Current value function
        
        # Initialize state space
        self.S = self._generate_state_space()
        
        # Initialize action space (edges from each node)
        self.A = self._generate_action_space()
        
    
    def _generate_state_space(self) -> Set[Tuple]:
        """
        Generate the state space S.
        A state is defined as the current node.
        
        Returns:
            Set of all possible states
        """
        S = set(self.env_graph.nodes())
        
        return S
    
    def _generate_action_space(self) -> Dict:
        """
        Generate action space for each node.
        Actions are edges that can be taken from each node.
        
        Returns:
            Dictionary mapping nodes to available actions (adjacent nodes)
        """
        A = {}
        for node in self.env_graph.nodes():
            # Actions are moving to adjacent nodes
            A[node] = list(self.env_graph.neighbors(node))
        return A
    
    def transition_function(self, s: Tuple, a: int, s_prime: Tuple) -> float:
        """
        Transition function T(s, a, s').
        
        Args:
            s: Current state (current_node)
            a: Action (next_node to move to)
            s_prime: Next state (next_node)
        
        Returns:
            Probability of transition (1.0 for deterministic, 0.0 for impossible)
        """
        current_node = s
        next_node = s_prime
        
        # Check if action is valid (edge exists)
        if a not in self.A.get(current_node, []):
            return 0.0
        
        # Check if next state is consistent with action
        if next_node != a:
            return 0.0
        
        return 1.0  # Deterministic transition
    
    def reward_function(self, state: Tuple, t: Optional[int] = None) -> float:
        """
        Compute reward R_t(s) for a given state.
        Uses the provided target_and_visibility_reward function.
        
        Args:
            state: Current state (current_node)
            t: Time step (optional, for time-dependent rewards)
        
        Returns:
            Reward value
        """

        # Use the provided reward function
        return target_and_visibility_reward(env_graph=self.env_graph, input_node=state, unreached_targets=self.unreached_targets)

    
    def value_iteration_convergence(self, initial_Value: Dict) -> Dict:
        """
        Perform value iteration until convergence (lines 3-12 of Algorithm 1).
        
        Returns:
            Converged value function
        """
        # Initialize value function (line 3)
        for s in self.S:
            self.V_t[s] = initial_Value[s]
        
        V_t_previous = self.V_t.copy()
        max_iteration = 1000
        iteration = 0
        delta = np.inf  # Line 5 
        
        while delta >= self.epsilon:  # Inner convergence loop (line 6)
            for s in self.S:  # Line 7
                
                # Store previous value (line 8)
                V_t_previous[s] = self.V_t[s]
                
                # Bellman update (line 9)
                if s not in self.A or len(self.A[s]) == 0:
                    self.V_t[s] = self.reward_function(s) # in case there's no available action on s
                else:
                    value_list=[]
                    max_value = -np.inf

                    for a in self.A[s]:
                        s_prime = a # Deterministic next state
                        trans_prob = self.transition_function(s, a, s_prime) # This will always be one for now
                        if trans_prob > 0:
                            value_list.append(
                                trans_prob * (
                                    self.reward_function(s_prime) + 
                                    self.gamma * self.V_t[s_prime]
                                )
                            )

                    max_value = max(value_list) if value_list else -np.inf
                    self.V_t[s] = max_value
            
            # Update delta (line 11)
            delta = max(abs(self.V_t[s] - V_t_previous[s]) for s in self.S)

            iteration += 1
            if iteration >= max_iteration:
                print("Warning: Value iteration did not converge within the maximum number of iterations.")
                break
        
        return self.V_t
    
    def extract_policy(self) -> Dict:
        """
        Extract optimal policy from value function.
        
        Returns:
            Policy mapping states to optimal actions
        """
        policy = {}
        
        for s in self.S:
            
            if s not in self.A or len(self.A[s]) == 0:
                policy[s] = None
                continue
            
            best_action = None
            best_value = -np.inf
            
            for a in self.A[s]:
                value = 0
                s_prime = a # Deterministic next state for now
                trans_prob = self.transition_function(s, a, s_prime)
                if trans_prob > 0:
                    value += trans_prob * (
                        self.reward_function(s_prime) + 
                        self.gamma * self.V_t[s_prime]
                    )
                
                if value > best_value:
                    best_value = value
                    best_action = a
            
            policy[s] = best_action
        
        return policy
    
    def run(self) -> List[Tuple]:
        """
        Run the complete Suboptimal Value Iteration Algorithm
        """
        # Line 2 and 3: Initialize the value function
        for s in self.S:
            self.V_t[s] = 0.0
        current_state = self.s0
        
        print(f"Initial unreached targets: {self.unreached_targets}")
        t = 0
        trajectory = []

        while len(self.unreached_targets) > 0:
            # Line 4: Perform value iteration to convergence
            self.V_t = self.value_iteration_convergence(initial_Value=self.V_t) 
            
            update_needed = False       

            # Line 13-18: Extract policy and execute
            while not update_needed: 

                # Line 14: Extract policy and execute
                policy = self.extract_policy()
                next_state = policy[current_state]

                if next_state is None:  # Handle case where no action available
                    print(f"Warning: No action available from state {current_state}")
                    break

                trajectory.append((current_state, next_state)) # append it as a tuple
                
                t += 1
                
                current_state = next_state

                # Line 17: Update set to be covered
                if next_state in self.unreached_targets:
                    self.env_graph.nodes[next_state]["type"] = "target_reached"
                    self.unreached_targets.remove(next_state)
                    update_needed = True
                    
                    # Subtract stored path contributions for this target
                    if "stored_path_contributions" in self.env_graph.nodes[next_state]:
                        for edge, weight in self.env_graph.nodes[next_state]["stored_path_contributions"]:
                            self.env_graph.edges[edge]['num_used'] -= weight
                        # Clear stored contributions after subtracting
                        self.env_graph.nodes[next_state]["stored_path_contributions"] = []

                # Update the visibility of the edges
                if "visible_edges" in self.env_graph.nodes[next_state]:
                    visible_edges = self.env_graph.nodes[next_state]["visible_edges"]
                    visible_unexplored_edges = [edge for edge in visible_edges 
                                            if self.env_graph.edges[edge].get("observed_edge", True) == False]
                    if len(visible_unexplored_edges) > 0:
                        for edge in visible_unexplored_edges:
                            self.env_graph.edges[edge]["observed_edge"] = True
                        update_needed = True

                print(f"Time step {t}: Moved to {current_state}, Remaining targets: {len(self.unreached_targets)}")

        return trajectory
    

class SuboptimalValueIteration:
    """
    Implementation of Suboptimal Value Iteration for multi-agent Steiner TSP
    with uncertain edges and visibility constraints.
    Modified to handle arbitrary edge lengths with 60Hz updates.
    """
    
    def __init__(self, env_graph: nx.Graph, s0, epsilon: float = 0.01, gamma: float = 0.1, sim_freq: int = 60):
        """
        Initialize the value iteration algorithm.
        
        Args:
            env_graph: NetworkX graph representing the environment
            s0: Initial state 
            epsilon: Convergence threshold for value iteration
            gamma: Discount factor
            sim_freq: Simulation frequency (default: 60Hz)
        """
        self.env_graph = env_graph
        self.s0 = s0
        self.epsilon = epsilon
        self.gamma = gamma
        self.dt = 1.0 / sim_freq  # Time step at specified frequency
        self.speed = 1.0  # Movement speed (units per second)

        self.unreached_targets = [node for node, data in self.env_graph.nodes(data=True) 
                                 if data.get("type") == "target_unreached"]
        
        # Initialize value functions
        self.V_t = {}  # Current value function
        
        # Initialize state space (now includes traversal state)
        self.S = self._generate_state_space()
        
        # Initialize action space (edges from each node)
        self.A = self._generate_action_space()
        
    
    def _generate_state_space(self) -> Set[Tuple]:
        """
        Generate the state space S.
        A state is defined as the current node (when at a node).
        
        Returns:
            Set of all possible states
        """
        S = set(self.env_graph.nodes())
        
        return S
    
    def _generate_action_space(self) -> Dict:
        """
        Generate action space for each node.
        Actions are edges that can be taken from each node.
        
        Returns:
            Dictionary mapping nodes to available actions (adjacent nodes)
        """
        A = {}
        for node in self.env_graph.nodes():
            # Actions are moving to adjacent nodes
            A[node] = list(self.env_graph.neighbors(node))
        return A
    
    def get_edge_length(self, node1, node2) -> float:
        """
        Get the length/weight of an edge between two nodes.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Edge length (defaults to 1.0 if not specified)
        """
        if self.env_graph.has_edge(node1, node2):
            return self.env_graph.edges[node1, node2].get("distance")
        return float('inf')
    
    def transition_function(self, s: Tuple, a: int, s_prime: Tuple) -> float:
        """
        Transition function T(s, a, s').
        
        Args:
            s: Current state (current_node)
            a: Action (next_node to move to)
            s_prime: Next state (next_node)
        
        Returns:
            Probability of transition (1.0 for deterministic, 0.0 for impossible)
        """
        current_node = s
        next_node = s_prime
        
        # Check if action is valid (edge exists)
        if a not in self.A.get(current_node, []):
            return 0.0
        
        # Check if next state is consistent with action
        if next_node != a:
            return 0.0
        
        return 1.0  # Deterministic transition
    
    def reward_function(self, state: Tuple, t: Optional[int] = None) -> float:
        """
        Compute reward R_t(s) for a given state.
        Uses the provided target_and_visibility_reward function.
        
        Args:
            state: Current state (current_node)
            t: Time step (optional, for time-dependent rewards)
        
        Returns:
            Reward value
        """

        # Use the provided reward function
        return target_and_visibility_reward(env_graph=self.env_graph, input_node=state, unreached_targets=self.unreached_targets)

    
    def value_iteration_convergence(self, initial_Value: Dict) -> Dict:
        """
        Perform value iteration until convergence (lines 3-12 of Algorithm 1).
        
        Returns:
            Converged value function
        """
        # Initialize value function (line 3)
        for s in self.S:
            self.V_t[s] = initial_Value[s]
        
        V_t_previous = self.V_t.copy()
        max_iteration = 1000
        iteration = 0
        delta = np.inf  # Line 5 
        
        while delta >= self.epsilon:  # Inner convergence loop (line 6)
            for s in self.S:  # Line 7
                
                # Store previous value (line 8)
                V_t_previous[s] = self.V_t[s]
                
                # Bellman update (line 9)
                if s not in self.A or len(self.A[s]) == 0:
                    self.V_t[s] = self.reward_function(s) # in case there's no available action on s
                else:
                    value_list=[]
                    max_value = -np.inf

                    for a in self.A[s]:
                        s_prime = a # Deterministic next state
                        trans_prob = self.transition_function(s, a, s_prime) # This will always be one for now
                        if trans_prob > 0:
                            # Get edge length for discounting
                            edge_length = self.get_edge_length(s, s_prime)
                            # Discount based on time to traverse edge
                            time_to_traverse = edge_length / self.speed
                            discount_factor = self.gamma ** time_to_traverse
                            
                            value_list.append(
                                trans_prob * (
                                    self.reward_function(s_prime) + 
                                    discount_factor * self.V_t[s_prime]
                                )
                            )

                    max_value = max(value_list) if value_list else -np.inf
                    self.V_t[s] = max_value
            
            # Update delta (line 11)
            delta = max(abs(self.V_t[s] - V_t_previous[s]) for s in self.S)

            iteration += 1
            if iteration >= max_iteration:
                print("Warning: Value iteration did not converge within the maximum number of iterations.")
                break
        
        return self.V_t
    
    def extract_policy(self) -> Dict:
        """
        Extract optimal policy from value function.
        
        Returns:
            Policy mapping states to optimal actions
        """
        policy = {}
        
        for s in self.S:
            
            if s not in self.A or len(self.A[s]) == 0:
                policy[s] = None
                continue
            
            best_action = None
            best_value = -np.inf
            
            for a in self.A[s]:
                value = 0
                s_prime = a # Deterministic next state for now
                trans_prob = self.transition_function(s, a, s_prime)
                if trans_prob > 0:
                    # Get edge length for discounting
                    edge_length = self.get_edge_length(s, s_prime)
                    time_to_traverse = edge_length / self.speed
                    discount_factor = self.gamma ** time_to_traverse
                    
                    value += trans_prob * (
                        self.reward_function(s_prime) + 
                        discount_factor * self.V_t[s_prime]
                    )
                
                if value > best_value:
                    best_value = value
                    best_action = a
            
            policy[s] = best_action
        
        return policy
    
    def run(self) -> List[Tuple]:
        """
        Run the complete Suboptimal Value Iteration Algorithm with 60Hz updates.
        """
        # Line 2 and 3: Initialize the value function
        for s in self.S:
            self.V_t[s] = 0.0
        current_state = self.s0
        
        print(f"Initial unreached targets: {self.unreached_targets}")
        t = 0
        real_time = 0.0  # Track real time in seconds
        trajectory = []
        
        # Traversal state
        is_traversing = False
        traversal_target = None
        traversal_start_time = 0.0
        traversal_duration = 0.0

        while len(self.unreached_targets) > 0:
            # Line 4: Perform value iteration to convergence
            if not is_traversing:
                self.V_t = self.value_iteration_convergence(initial_Value=self.V_t) 
            
            update_needed = False       

            # Line 13-18: Extract policy and execute
            while not update_needed: 
                
                if not is_traversing:
                    # Line 14: Extract policy and determine next action
                    policy = self.extract_policy()
                    next_state = policy[current_state]

                    if next_state is None:  # Handle case where no action available
                        print(f"Warning: No action available from state {current_state}")
                        break

                    # Start traversing edge
                    edge_length = self.get_edge_length(current_state, next_state)
                    traversal_duration = edge_length / self.speed
                    traversal_start_time = real_time
                    traversal_target = next_state
                    is_traversing = True
                    
                    print(f"Time {real_time:.3f}s (step {t}): Starting traversal from {current_state} to {next_state} (distance: {edge_length:.2f}, duration: {traversal_duration:.3f}s)")
                
                # Update at 60Hz
                t += 1
                real_time += self.dt
                
                # Check if we've arrived at the target node
                if is_traversing and (real_time - traversal_start_time) >= traversal_duration:
                    # Arrival!
                    trajectory.append((current_state, traversal_target))
                    current_state = traversal_target
                    is_traversing = False
                    
                    print(f"Time {real_time:.3f}s (step {t}): Arrived at {current_state}")

                    # Line 17: Update set to be covered
                    if current_state in self.unreached_targets:
                        self.env_graph.nodes[current_state]["type"] = "target_reached"
                        self.unreached_targets.remove(current_state)
                        update_needed = True
                        
                        # Subtract stored path contributions for this target
                        if "stored_path_contributions" in self.env_graph.nodes[current_state]:
                            for edge, weight in self.env_graph.nodes[current_state]["stored_path_contributions"]:
                                self.env_graph.edges[edge]['num_used'] -= weight
                            # Clear stored contributions after subtracting
                            self.env_graph.nodes[current_state]["stored_path_contributions"] = []

                    # Update the visibility of the edges
                    if "visible_edges" in self.env_graph.nodes[current_state]:
                        visible_edges = self.env_graph.nodes[current_state]["visible_edges"]
                        visible_unexplored_edges = [edge for edge in visible_edges 
                                                if self.env_graph.edges[edge].get("observed_edge", True) == False]
                        if len(visible_unexplored_edges) > 0:
                            for edge in visible_unexplored_edges:
                                self.env_graph.edges[edge]["observed_edge"] = True
                            update_needed = True

                    print(f"Remaining targets: {len(self.unreached_targets)}")
                
                # Break if still traversing (wait for arrival)
                if is_traversing and not update_needed:
                    break

        return trajectory