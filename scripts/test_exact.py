import unittest as ut, networkx as nx, numpy as np, exact, cupy as cp


class TestNodeToMacro(ut.TestCase):
    def setUp(self):
        beta = 0.1
        self.g = nx.path_graph(4)
        # self.g = nx.krackhardt_kite_graph()
        n = len(self.g)
        self.g = nx.path_graph(n)
        self.A = cp.asarray(nx.adjacency_matrix(self.g).todense())

        self.states, self.allowed = exact.gen_states(n)
        E = exact.ising(self.states, self.A)
        p, p0 = exact.get_transfer(n, E, beta, self.allowed, self.states)
        self.model = exact.NodeToMacroBit(p, p0, self.states)

    def test_setup(self):
        print("testing setup")
        self.model.setup(ss=self.allowed)
        # shape should be 3 x n_nodes x 2 states (for ising)
        s = self.model.transfer_matrix.shape
        # The macrobit has three states
        self.assertEqual(s[0], 3)
        s = self.model.evolve_states.shape
        self.assertEqual(s[0], 3)
        self.assertEqual(s[1], len(self.g))
        self.assertEqual(s[2], 2)  # for ising

        # check transfer xi -> xj, rows should sum up to 1
        rows = np.allclose(self.model.transfer_matrix.sum(1), 1)
        print(rows, self.model.transfer_matrix.sum(1))
        print(self.model.transfer_matrix.sum(0))
        print(self.model.transfer_matrix)
        self.assertTrue(np.all(rows))

        import proplot as plt

        fig, ax = plt.subplots()
        h = ax.imshow(self.model.transfer_matrix.get())
        ax.colorbar(h)
        fig.show()
        plt.show(block=1)
