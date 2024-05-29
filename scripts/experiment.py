import networkx as nx, numpy as np


def small_tree() -> nx.Graph:
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)

    g.add_edge(1, 4)
    g.add_edge(1, 5)

    g.add_edge(2, 6)
    g.add_edge(2, 7)

    g.add_edge(3, 8)
    g.add_edge(3, 9)
    return g


def small_tree_cross() -> nx.Graph:
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)

    g.add_edge(1, 4)
    g.add_edge(1, 5)

    g.add_edge(2, 6)
    g.add_edge(2, 7)

    g.add_edge(3, 8)
    g.add_edge(3, 9)

    # cross
    g.add_edge(2, 3)
    return g


def system_14() -> nx.Graph:
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(3, 4)
    g.add_edge(3, 5)

    g.add_edge(1, 6)
    g.add_edge(1, 7)
    g.add_edge(1, 8)
    g.add_edge(1, 9)
    g.add_edge(1, 2)
    return g


def double_circle() -> nx.Graph:
    g = nx.path_graph(5)
    g = nx.cycle_graph(5)
    for i in range(3):
        g.add_edge(5 + i, 5 + i + 1)
    g.add_edge(0, 5)
    g.add_edge(8, 0)

    # fig, ax = plt.subplots()
    # nx.draw(g, with_labels = 1)
    # plt.show(block = 1)
    return g


def cycle10() -> nx.Graph:
    return nx.cycle_graph(10)


def y_split():
    g = nx.path_graph(9)
    g.add_edge(7, 9)
    return g
