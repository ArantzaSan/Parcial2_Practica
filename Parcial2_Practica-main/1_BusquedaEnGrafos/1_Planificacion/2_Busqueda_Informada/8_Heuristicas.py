import heapq
from collections import namedtuple

# Representation of a node with cost and path
NodeInfo = namedtuple('NodeInfo', ['total_cost', 'state', 'path'])

def estimate_remaining_cost(current, target):
    # Manhattan heuristic to estimate the remaining distance
    return abs(current[0] - target[0]) + abs(current[1] - target[1])

def a_star_pathfinder(graph_structure, start_node, end_node):
    # Priority queue to explore nodes, ordered by estimated total cost
    open_set = [NodeInfo(estimate_remaining_cost(start_node, end_node), start_node, [start_node])]
    # Record of the minimum known cost to reach each node
    accumulated_costs = {start_node: 0}
    # Set of nodes already visited to avoid cycles
    visited_nodes = set()

    while open_set:
        # Get the node with the lowest estimated total cost
        current_node_info = heapq.heappop(open_set)
        current_cost = current_node_info.total_cost
        current_state = current_node_info.state
        current_path = current_node_info.path

        # If the target is reached, return the found path
        if current_state == end_node:
            return current_path

        # Mark the current node as visited
        visited_nodes.add(current_state)

        # Explore the neighbors of the current node
        for neighbor, transition_cost in graph_structure.get(current_state, []):
            if neighbor not in visited_nodes:
                new_cost = accumulated_costs[current_state] + transition_cost
                if neighbor not in accumulated_costs or new_cost < accumulated_costs[neighbor]:
                    accumulated_costs[neighbor] = new_cost
                    priority = new_cost + estimate_remaining_cost(neighbor, end_node)
                    new_path = current_path + [neighbor]
                    heapq.heappush(open_set, NodeInfo(priority, neighbor, new_path))

    # If the open set becomes empty without finding the target
    return None

# Example graph represented as a dictionary of lists of tuples (neighbor, cost)
map_data = {
    (0, 0): [((1, 0), 1), ((0, 1), 1)],
    (1, 0): [((0, 0), 1), ((1, 1), 1)],
    (0, 1): [((0, 0), 1), ((1, 1), 1)],
    (1, 1): [((1, 0), 1), ((0, 1), 1), ((1, 2), 1)],
    (1, 2): [((1, 1), 1)]
}

# Execute the A* search from node (0, 0) to node (1, 2)
route = a_star_pathfinder(map_data, (0, 0), (1, 2))
print(f"Path from (0, 0) to (1, 2) using A*: {route}")
