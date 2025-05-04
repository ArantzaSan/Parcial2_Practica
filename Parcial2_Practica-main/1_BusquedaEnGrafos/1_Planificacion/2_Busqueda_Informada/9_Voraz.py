import heapq
from collections import namedtuple

# Representation of a node with its estimate and the path
NodeInfo = namedtuple('NodeInfo', ['estimation', 'state', 'route'])

def calculate_heuristic(current, target):
    # Manhattan distance heuristic
    return abs(current[0] - target[0]) + abs(current[1] - target[1])

def greedy_best_first_search(graph_structure, start_node, goal_node):
    # Priority queue for nodes to explore, prioritized by heuristic
    open_set = [NodeInfo(calculate_heuristic(start_node, goal_node), start_node, [start_node])]
    # Set of already examined states
    explored_nodes = set()

    while open_set:
        # Get the node with the lowest heuristic estimate
        current_node_info = heapq.heappop(open_set)
        _, current_state, current_route = current_node_info

        # If the goal is reached, return the path
        if current_state == goal_node:
            return current_route

        # If the current state has not been explored
        if current_state not in explored_nodes:
            explored_nodes.add(current_state)

            # Examine the successors of the current state
            for successor, _ in graph_structure.get(current_state, []):
                if successor not in explored_nodes:
                    new_route = current_route + [successor]
                    priority = calculate_heuristic(successor, goal_node)
                    heapq.heappush(open_set, NodeInfo(priority, successor, new_route))

    # If the open set becomes empty without finding the goal
    return None

# Example graph as a dictionary of lists of tuples (neighbor, cost)
node_network = {
    (0, 0): [((1, 0), 1), ((0, 1), 1)],
    (1, 0): [((0, 0), 1), ((1, 1), 1)],
    (0, 1): [((0, 0), 1), ((1, 1), 1)],
    (1, 1): [((1, 0), 1), ((0, 1), 1), ((1, 2), 1)],
    (1, 2): [((1, 1), 1)]
}

# Execute Greedy Best-First Search from node (0, 0) to node (1, 2)
path = greedy_best_first_search(node_network, (0, 0), (1, 2))
print(f"Path from (0, 0) to (1, 2) using Greedy Best-First Search: {path}")
