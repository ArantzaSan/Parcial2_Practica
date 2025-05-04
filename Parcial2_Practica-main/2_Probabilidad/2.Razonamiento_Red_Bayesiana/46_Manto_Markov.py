import networkx as nx
import numpy as np

def create_markov_blanket(graph, node):
    """
    Function to compute the Markov Blanket of a given node in a Bayesian Network.
    :param graph: A directed graph (Bayesian Network) using NetworkX.
    :param node: The node for which the Markov Blanket is computed.
    :return: A set containing the Markov Blanket of the node.
    """
    parents = set(graph.predecessors(node))
    children = set(graph.successors(node))
    co_parents = set()

    for child in children:
        co_parents.update(graph.predecessors(child))

    markov_blanket = parents | children | (co_parents - {node})
    return markov_blanket

# Example Bayesian Network
bayesian_network = nx.DiGraph()
bayesian_network.add_edges_from([
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'D'),
    ('C', 'D'),
    ('D', 'E')
])

# Node for which we want the Markov Blanket
target_node = 'D'
markov_blanket = create_markov_blanket(bayesian_network, target_node)

print(f"Markov Blanket of node {target_node}: {markov_blanket}")