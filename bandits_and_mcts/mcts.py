import numpy as np
import numpy.random as npr
from collections import defaultdict

def env_model(env):
    """
    Returns method to find next state for a given state, action pair
    """
    return lambda s, a: env.P[s][a][0]
class MCTS:

    def __init__(self, env):
        self.env = env
        
        # Num visits, num wins
        self.state_counts = np.zeros(env.nS) + 0.000001
        self.state_values = np.zeros(env.nS)
        self.model = env_model(env)
    
    def value_fcn(self):
        values = self.state_values / self.state_counts
        # values = self.state_counts
        return values.reshape((self.env.nrow, self.env.ncol))
    
    def visitation_fcn(self):
        return self.state_counts.reshape(self.env.nrow, self.env.ncol)
    
    def next_move(self, state, max_iters=5000):
        top_node = Node(state, None, False, 0, self.model)
        for _ in range(max_iters):
            self.step(top_node)
        
        return self._best_action(top_node)

    def step(self, node):
        # print(f"Current state: {node}")
        next_node = self._tree_policy(node)

        # print(f"Simulating from: {next_node}")
        value = self._simulation(next_node.state)
        
        # print(f"Backpropping value {value}")
        self._backprop(next_node, value)

    
    def _tree_policy(self, current_node):
        """Select or create a leaf node from the nodes already contained within the search tree
        """

        while not current_node.is_terminal:
            if (current_node.fully_expanded()):
                current_node = self._ucb_select(current_node)
            else:
                return self._expand(current_node)
        
        return current_node

    def _expand(self, node):
        action = npr.choice(node.actions_left)
        return node.expand(action, self.model)

    def _ucb_select(self, current_node):
        children = current_node.children()       
        states = [child.state for child in children]
        counts = self.state_counts[states]
        ucb = np.sqrt(2 * np.log(counts.sum() / counts))
        avgs = self.state_values[states] / self.state_counts[states]
        
        return children[np.argmax(avgs + ucb)]
    
    def _mean_reward(self, node):
        if self.state_counts[node.state] == 0:
            return 0
        return self.state_values[node.state] / self.state_counts[node.state]
    
    def _simulation(self, state):
        """Simulates to the bottom of the tree randomly from this state.
        """

        is_terminal = False
        while not is_terminal:
            action = self.env.action_space.sample()
            _, state, reward, is_terminal = self.model(state, action)

        return reward
    
    def _backprop(self, node, value):
        """Use parents of this node to propagate the value
        """
        while node.parent != None:
            # print(f"Backpropping {node}")
            self.state_counts[node.state] += 1
            self.state_values[node.state] += value
            
            node = node.parent
        # print("")

    def _best_action(self, node):
        child_states, bad_actions = node.child_states()

        if self.env.goal_state() in child_states:
            return child_states.index(self.env.goal_state())
        child_values = self.state_values[child_states]
        child_visits = self.state_counts[child_states]
        
        values = child_values / child_visits
        values[bad_actions] = 0

        action = np.argmax(values)
        # print(f"Action values: {values}") 
        print(f"Move {self.env.get_action(action)}")
        return np.argmax(values)


class Node:

    def __init__(self, state, parent, is_terminal, reward, model):
        self.state = state
        self.parent = parent
        self.is_terminal = is_terminal
        self.reward = reward
        self.expanded = {}
        self.actions_left = list(range(4))
        self.model = model
    
    def __str__(self) -> str:
        return f"({self.state}, {self.is_terminal}, {len(self.expanded)})"

    def children(self):
        return [self.expanded[action] for action in sorted(self.expanded)]
    
    def child_states(self):
        
        children = []
        bad_actions = []
        for a in range(4):
            ns = self.model(self.state, a)[1]
            if ns == self.state:
                bad_actions.append(a)
            
            children.append(ns)

        return children, bad_actions
            

    def fully_expanded(self):
        child_states = {self.model(self.state, a)[1] for a in range(4)} 
        return len(self.expanded) == len(child_states)

    def expand(self, action, model):
        # print(self.actions_left)   
        self.actions_left.remove(action)
        _, next_state, reward, is_terminal = model(self.state, action)
        child = Node(next_state, self, is_terminal, reward, model)
        self.expanded[action] = child
        
        return child
        

        
        
        
        