import networkx as nx
import numpy as np

def calcular_manta_markov(grafo, nodo):
    """
    Función para calcular la Manta de Markov de un nodo dado en una Red Bayesiana.
    :param grafo: Un grafo dirigido (Red Bayesiana) usando NetworkX.
    :param nodo: El nodo para el cual se calcula la Manta de Markov.
    :return: Un conjunto que contiene la Manta de Markov del nodo.
    """
    padres = set(grafo.predecessors(nodo))
    hijos = set(grafo.successors(nodo))
    copadres = set()

    for hijo in hijos:
        copadres.update(grafo.predecessors(hijo))

    manta_markov = padres | hijos | (copadres - {nodo})
    return manta_markov

# Ejemplo de Red Bayesiana
red_bayesiana = nx.DiGraph()
red_bayesiana.add_edges_from([
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'D'),
    ('C', 'D'),
    ('D', 'E')
])

# Nodo para el cual queremos la Manta de Markov
nodo_objetivo = 'D'
manta = calcular_manta_markov(red_bayesiana, nodo_objetivo)

print(f"Manta de Markov del nodo {nodo_objetivo}: {manta}")
