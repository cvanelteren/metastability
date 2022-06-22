import cmasher as cmr, pandas as pd, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
from tqdm import tqdm

try:
    import cupy as np

    np.eye(5)
    print("Using cupy")

except:
    print("Using numpy")
    import numpy as np
# import numpy as np


def gen_binary(idx, n):
    return np.asarray([int(i) for i in format(idx, f"0{n}b")])


def to_binary(states):
    idx = 0
    for kdx, i in enumerate(states[::-1]):
        idx += 2**kdx * i
    return int(idx)


def gen_states(n) -> tuple:
    """
    Creates binary state space and returns allowed transitions
    """
    states = np.zeros((2**n, n))
    allowed = {}
    # ntrans = {}
    from tqdm import tqdm
    from copy import deepcopy

    for idx in tqdm(range(2**n)):
        states[idx] = gen_binary(idx, n)
        for node in range(n):
            state = states[idx].copy()
            state[node] = (state[node] + 1) % 2
            diff = abs(state - states[idx]).sum()
            assert diff == 1, diff
            kdx = to_binary(state)
            allowed[idx] = allowed.get(idx, []) + [kdx]
    return states, allowed


def get_transfer(n, E, beta, allowed, states, node=-1, eta=0) -> tuple:
    """
    Construct transfer matrix based on energy and temperature
    """
    p = np.zeros((2**n, 2**n))
    p0 = np.zeros(2**n)
    # beta = 0.5
    # sanity checks
    assert len(allowed) == 2**n
    pairs = []
    for state, trans in allowed.items():
        e1 = E[state]
        p0[state] = np.exp(-beta * e1.sum())

        # sanity checks
        assert np.isnan(p0[state]) == False, (p0[state], np.exp(-beta * e1.sum()))
        assert p0[state] >= 0, f"{p0[state]} {e1.sum()} {e1}"
        for other in trans:
            cidx = np.where(states[state] - states[other] != 0)[0]
            assert len(cidx) == 1

            e1 = E[state, cidx]
            e2 = E[other, cidx]

            # if abs(eta) > 0 and node == cidx:
            #     if states[state, cidx] == 0:
            #         e1 = -np.inf
            #         if states[other, cidx] == 1:
            #             e2 = np.inf
            #         else:
            #             e2 = e1
            #     elif states[state, cidx] == 1:
            #         e1 = np.inf
            #         if states[other, cidx] == 0:
            #             e2 = -np.inf
            #         else:
            #             e2 = e1

            delta = e2 - e1
            # if np.isnan(delta):
            # pi = 0
            # else:
            # pi = 1 / n * 1 / (1 + np.exp(beta * delta))

            pi = 1 / n * 1 / (1 + np.exp(beta * delta))

            # sanity checks
            assert 0 <= pi <= 1, f"{delta=} {e2.sum()=} {e1.sum()=}"
            p[state, other] = pi
            if (state, other) in pairs:
                assert False, pairs
            pairs.append((state, other))
            # p[other, state] = pi
            if state == other:
                assert 0, "States cannot equal"

    np.fill_diagonal(p, 1 - p.sum(1))
    assert np.any(np.isnan(p)) == False, p
    from scipy import linalg
    from numpy import argmax

    # e, v = linalg.eig(p.get().T)
    # idx = argmax(e)
    # p0 = v[:, idx]
    p0 /= p0.sum()
    p0 = np.asarray(p0)
    # p0 = np.array(p0)
    # p0[p0 == np.inf] = 1
    # for pi in p0:
    # print(pi)
    return p, p0


def get_allowed_per_mag(states, p_state):
    s = np.mean(states, 1)
    n = states.shape[1]
    bins = np.linspace(0.0 - 1 / (2 * (n)), 1.0 + 1 / (2 * (n)), n + 2)
    vals = np.zeros(bins.size)
    idx = np.digitize(s, bins)
    allowed_per_mag = {}
    for jdx, bin in enumerate(idx):
        vals[bin] += p_state[jdx]
        mag = float(bins[bin])
        allowed_per_mag[mag] = allowed_per_mag.get(mag, []) + [jdx]
    return allowed_per_mag, bins, vals


def entropy(p):
    return -np.nansum(p * np.log2(p), axis=-1)


class Data:
    def __init__(
        self,
        mi_system_node,
        entropy_system_node,
        c_entropy_system_node,
        mi_node_system,
        entropy_node_system,
        c_entropy_node_system,
        avg_node_energy,
        transfer_matrix,
        mag,
    ):
        for name, d in locals():
            store_numpy_or_cuda(name, d)

    def store_numpy_or_cuda(self, name, d):
        # checks if data is ndarray or cuda ndarray
        if hasattr(d, "get"):
            self.__dict__[name] = d.get()
        else:
            self.__dict__[name] = d


from dataclasses import dataclass


@dataclass
class AbstractSimulator:
    transfer_matrix: np.ndarray  # p: 2**n x 2**n
    p_state: np.ndarray  # eq. dist p0: 2**n
    states: np.ndarray  # states 2**n x n
    allowed = []

    def setup(self):
        """
        Create out of equilibrium buffer for state evolution
        """
        assert 0

    def update(self, *args, **kwargs):
        """
        Compute entropy, conditional entropy, and mutual information
        for 1 time step update
        """
        assert 0

    def adjust_allowed(self):
        # only simulate states with mag == x
        if self.allowed:
            n = self.states.shape[1]
            idx = np.array(
                [i for i in range(2**n) if i not in self.allowed], dtype=int
            )
            # dont allow othe states
            self.evolve_states[idx] = 0
            self.p_state[idx] = 0
            self.p_state /= self.p_state.sum()

            # prevent zero division in magnetization match case
            self.p_state[np.isnan(self.p_state)] = 0
            assert np.allclose(np.sum(self.p_state), 1)
            print(f"Removed {2**n - idx.size} states")

        self.evolve_states /= self.evolve_states.sum(0)[None]
        self.evolve_states[np.isnan(self.evolve_states)] = 0


class NodeToSystem(AbstractSimulator):
    """
    Computes I(S^t : s_i)
    """

    p_node = np.empty(1)  # create in setup

    def setup(self, allowed=[]):
        self.allowed = allowed
        n = self.states.shape[1]  # should be 2**n x n
        # P(S^t : s_i)
        self.evolve_states = np.zeros((2**n, n, 2))
        for idx, s in enumerate(self.states):
            for node, si in enumerate(s):
                self.evolve_states[idx, node, int(si)] += self.p_state[idx]

        self.adjust_allowed()
        self.p_node = np.zeros((n, 2))
        self.p_node[..., 1] = self.p_state @ self.states
        self.p_node[..., 0] = 1 - self.p_node[..., 1]

        # assert np.all(np.allclose(self.p_node, 0.5))

    def update(self):
        # H(S^t)
        pst = np.einsum("ij, kij -> ki", self.p_node, self.evolve_states)
        H = entropy(pst.T)
        # conditional entropy
        # H(S^t | s_i) = - \sum p(s_i ) \sum_S^t p(S^t | s ,_i) log2(p(S^t | s_i))
        tmp = entropy(self.evolve_states.T).T
        pc = self.evolve_states.copy()
        tmp[np.isnan(tmp)] = 0
        HC = np.einsum("ij, ij-> i", self.p_node, tmp)
        self.evolve()
        # return H, HC, H - HC
        return H, HC, H - HC, pst, pc

    def evolve(self):
        self.evolve_states = np.einsum(
            "ijk, il -> ljk",
            self.evolve_states,
            self.transfer_matrix,
        )
        return self.evolve_states


class NodeToMacroBit(NodeToSystem):
    def setup(self, allowed=[]):
        # setup parent class
        super(NodeToMacroBit, self).setup(allowed)

        # add functionality of the macrobit
        # find those states that fall within -1, 0, 1 sign
        state_signs = {}
        for idx, s in enumerate(self.states):
            sign = int(np.sign(s.mean() - 0.5))
            state_signs[sign] = state_signs.get(sign, []) + [idx]
        self.state_signs = state_signs  # maps sign to state_idx
        print(f"Found signs {state_signs.keys()}")

        # shorthands for clarity
        n_states_macrobit = len(state_signs)
        n_nodes = self.states.shape[1]

        self.p_macrobit = np.zeros((n_states_macrobit, n_nodes))
        self.p_macrobit_node = np.zeros(
            (n_states_macrobit, n_nodes, 2)
        )  # p(macrobit | node state) 3xnx2

    def update(self):
        # p(S^t)
        p_st = np.einsum("ij, kij -> ki", self.p_node, self.evolve_states)
        # convert to macrobit
        self.p_macrobit.fill(0)
        self.p_macrobit_node.fill(0)
        for idx, (sign, idxs) in enumerate(self.state_signs.items()):
            idxs = np.array(idxs, dtype=int)
            self.p_macrobit[idx] = p_st[idxs].sum(0)
            self.p_macrobit_node[idx] = self.evolve_states[idxs].sum(0)
        # renormalize conditional distribution
        self.p_macrobit /= self.p_macrobit.sum(0)[None]
        # self.p_macrobit_node /= self.p_macrobit_node.sum(0)[None]
        print(self.p_macrobit_node[:, 0, 1])
        print(self.evolve_states[:, 0, 0])
        print(self.evolve_states[:, 0, 1])
        print(self.evolve_states.sum(0))

        # convert 0 division to 0
        self.p_macrobit = np.nan_to_num(self.p_macrobit)
        self.p_macrobit_node = np.nan_to_num(self.p_macrobit_node)
        assert np.allclose(self.p_macrobit.sum(0), 1), self.p_macrobit.sum()
        assert np.allclose(
            self.p_macrobit_node.sum(0)[..., 0], 1
        ), self.p_macrobit_node.sum(0)[..., 0]

        assert np.allclose(
            self.p_macrobit_node.sum(0)[..., 1], 1
        ), self.p_macrobit_node.sum(0)[..., 1]

        H = entropy(self.p_macrobit.T)
        tmp = entropy(self.p_macrobit_node.T).T
        tmp = np.nan_to_num(tmp)
        HC = np.einsum("ij, ij-> i", self.p_node, tmp)
        self.evolve()
        return H, HC, H - HC


class SystemToNode(AbstractSimulator):
    """
    Computes I(s_i^t : S)
    """

    # reshape in setup
    conditional_node_state = np.empty(1)  # P(s_i^t | S)
    node_state = np.empty(1)  # P(s^t)

    def setup(self, allowed=[]):
        self.allowed = allowed
        n = self.states.shape[1]  # should be 2**n x n
        self.evolve_states = np.eye(2**n)
        self.adjust_allowed()
        self.conditional_node_state = np.zeros((2**n, n, 2))
        self.node_state = np.zeros((n, 2))

    def evolve(self):
        # compute the error
        self.evolve_states = self.evolve_states @ self.transfer_matrix
        return self.evolve_states

    def update(self):
        # Compute P(s_i^t | S)
        self.conditional_node_state[..., 1] = self.evolve_states.dot(self.states)
        self.conditional_node_state[..., 0] = 1 - self.conditional_node_state[..., 1]
        # Compute P(s_i^t)
        self.node_state[..., 1] = self.p_state @ self.conditional_node_state[..., 1]
        self.node_state[..., 0] = 1 - self.node_state[..., 1]

        # assert np.all(np.isclose(node_state.sum(-1), 1)), node_state.sum(-1)
        # assert np.all(np.allclose(np.sum(NS, -1), 1))

        H = entropy(self.node_state)
        HC = self.p_state @ entropy(self.conditional_node_state)
        # I[i] = H.sum() - HC.sum()

        # system  information decay I(S^t : S)
        h = entropy(self.p_state @ self.evolve_states)
        hc = self.p_state @ entropy(self.evolve_states)

        self.evolve()
        return H, HC, H - HC


@dataclass
class Settings:
    beta: float
    steps: int
    g: nx.Graph
    model: AbstractSimulator


class MetaDataFrame(pd.DataFrame):
    # store graph, magnetizations and prob of that mag occuring
    _metadata = ["mag_bin", "p_mag", "transfer_matrix", "p_state", "settings"]

    @property
    def _constructor(self):
        return self.__class__

    def __init__(
        self,
        *args,
        mag_bin=None,
        p_mag=None,
        transfer_matrix=None,
        p_state=None,
        settings=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mag_bin = mag_bin
        self.p_mag = p_mag
        self.transfer_matrix = transfer_matrix
        self.p_state = p_state
        self.settings = settings

    def __deepcopy__(self):
        print("deep copying")
        tmp = super().__deepcopy__(self)
        tmp._metadata = self._metadata
        for prop in self._metadata:
            tmp.__dict__[prop] = getattr(self, prop)


def ising(states, A, node=None, eta=None):
    return -np.einsum("ij, ij->ij", (states * 2 - 1), (states * 2 - 1).dot(A))


def sis(states, A):
    eta = 0.1  # infection rate
    mu = 0.1  # recovery rate
    rec = (states - 1) * (1 - (1 - eta) ** (states @ A))
    inf = -mu * states
    return inf - rec


def bornholdt(states, A, alpha=1.0):
    tmp = states * 2 - 1
    E = -np.einsum("ij, ij -> ij", tmp, tmp @ A)
    system = -(tmp * abs(tmp.mean(-1)[:, None])) * alpha
    return E + system


def node_involved_in(node, eta, p, states, E, beta, allowed):
    n = states.shape[1]

    for xi in allowed:
        for xj in allowed[xi]:
            if states[xi, node] != states[xj, node]:
                sign = 1
                if states[xi, node] == 0:
                    sign = -1

                # r = p[xi, xj] - eta / n
                # r = np.clip(r, 0, 1 / n)
                # p[xi, xj] += sign * eta / n
                # p[xi, xj] = np.clip(p[xi, xj], 0, 1)

                # p[xi, xi] += -sign * eta / n
                # p[xi, xi] = np.clip(p[xi, xi], 0, 1)

                e1 = E[xi, node] + sign * eta
                e2 = E[xj, node] - sign * eta
                delta = e2 - e1
                p[xi, xj] = 0
                if not np.isnan(delta):
                    p[xi, xj] = 1 / (n * (1 + np.exp(delta * beta)))

                # p[kdx, idx] += -sign * eta / n
                # p[kdx, idx] = np.clip(p[kdx, idx], 0, 1)

                # p[kdx, kdx] -= -sign * eta / n
                # p[kdx, kdx] = np.clip(p[kdx, kdx], 0, 1)

                # p[kdx, idx] += eta / n
                # p[kdx, idx] = np.clip(p[kdx, idx], 0, 1)
    np.fill_diagonal(p, 0)
    np.fill_diagonal(p, 1 - p.sum(1))
    # p /= p.sum(1)[:, None]
    assert np.all(np.allclose(p.sum(1), 1)), p.sum(1)


def experiment5_intervention(
    node, eta, p, p0, states, E, beta, settings, allowed, reduction
):
    from copy import deepcopy

    n = len(settings.g)
    m = settings.model(deepcopy(p), deepcopy(p0), states)

    m.setup(reduction)

    if node != -1:
        node_involved_in(node, eta, m.transfer_matrix, states, E, beta, allowed)

    from numpy import linalg, log, isnan, argmax

    # produces better numerical results than numpy
    from scipy import linalg as scalg

    transfer = m.transfer_matrix.get().T
    # e, v = linalg.eig(transfer)
    e, v = scalg.eig(transfer)
    largest_idx = argmax(e)
    largest = v[:, largest_idx]
    largest = abs(largest)
    largest /= largest.sum()

    m = settings.model(deepcopy(p), np.array(deepcopy(largest)), states)
    m.setup(reduction)

    p_states = np.zeros((settings.steps, len(states)))

    d = m.p_state

    d = 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(d) - np.sqrt(p0))
    print("difference", d)
    print(node, eta)
    p_states[0, :] = m.p_state.copy()
    for ti in range(1, settings.steps):
        p_states[ti] = m.p_state @ m.evolve_states
        m.evolve()

    return dict(
        node=node,
        eta=float(eta),
        ps=p_states.get().copy(),
        p0=m.p_state.get().copy(),
        largest=largest,
    )


def intervene(node, eta, p, p0, states, E, beta, settings, allowed, reduction):
    from copy import deepcopy

    n = len(settings.g)
    m = settings.model(deepcopy(p), deepcopy(p0), states)

    m.setup(reduction)

    if node != -1:
        node_involved_in(node, eta, m.transfer_matrix, states, E, beta, allowed)

    from numpy import linalg, log, isnan, argmax

    # produces better numerical results than numpy
    from scipy import linalg as scalg

    transfer = m.transfer_matrix.get().T
    # e, v = linalg.eig(transfer)
    e, v = scalg.eig(transfer)
    largest_idx = argmax(e)
    largest = v[:, largest_idx]
    largest = abs(largest)
    largest /= largest.sum()

    p_states = np.zeros((settings.steps, len(states)))
    print(node, eta)
    p_states[0, :] = m.p_state.copy()
    for ti in range(1, settings.steps):
        p_states[ti] = m.p_state @ m.evolve_states
        if ti == settings.steps // 2:
            m.transfer_matrix = p
        m.evolve()

    return dict(
        node=node,
        eta=float(eta),
        ps=p_states.get().copy(),
        p0=m.p_state.get().copy(),
        largest=largest,
    )


def sim2(
    settings,
    structure=None,
    e_func=ising,
    interventions=[],
    targets=[],
):
    # Compute energy and get transer matrix for temperature
    n = len(settings.g)
    if structure is None:
        A = np.asarray(
            nx.adjacency_matrix(
                settings.g,
                weight="weight",
            ).todense()
        )
        states, allowed = gen_states(n)
    else:
        A, states, allowed = structure

    # m.evolve_states = m.evolve_states[[0], :]
    E = e_func(states, A)
    p, p0 = get_transfer(n, E, settings.beta, allowed, states, -1, 0)
    allowed_starting, mags, p_mag = get_allowed_per_mag(states, p0)
    # extract allowed states
    #
    print(states)

    df = []
    from copy import deepcopy

    reduction = []
    for idx, s in enumerate(states):
        for prop in targets:
            if np.mean(s) == prop:
                reduction.append(idx)

    n = states.shape[1]

    from copy import deepcopy

    d0 = p0.copy()
    for intervention in interventions:
        intervention = {node: intervention for node in settings.g.nodes()}
        for idx, (node, eta) in enumerate(tqdm(intervention.items())):
            # p, p0 = get_transfer(n, E, settings.beta, allowed, states, node, eta)
            row = intervene(
                node,
                eta,
                deepcopy(p),
                deepcopy(p0),
                states,
                E,
                settings.beta,
                settings,
                allowed,
                reduction,
            )
            df.append(row)

    # control condition
    # p, p0 = get_transfer(n, E, settings.beta, allowed, state -1, 0)
    df.append(
        intervene(
            -1,
            0,
            deepcopy(p),
            deepcopy(p0),
            states,
            E,
            settings.beta,
            settings,
            allowed,
            reduction,
        )
    )
    return pd.DataFrame(df)


def experiment5(
    settings,
    structure=None,
    e_func=ising,
    interventions=[],
    targets=[],
):
    # Compute energy and get transer matrix for temperature
    n = len(settings.g)
    if structure is None:
        A = np.asarray(
            nx.adjacency_matrix(
                settings.g,
                weight="weight",
            ).todense()
        )
        states, allowed = gen_states(n)
    else:
        A, states, allowed = structure

    # m.evolve_states = m.evolve_states[[0], :]
    E = e_func(states, A)
    p, p0 = get_transfer(n, E, settings.beta, allowed, states, -1, 0)
    allowed_starting, mags, p_mag = get_allowed_per_mag(states, p0)
    # extract allowed states
    #
    df = []
    from copy import deepcopy

    reduction = []
    for idx, s in enumerate(states):
        for prop in targets:
            if np.mean(s) == prop:
                reduction.append(idx)

    n = states.shape[1]

    from copy import deepcopy

    d0 = p0.copy()
    for intervention in interventions:
        intervention = {node: intervention for node in settings.g.nodes()}
        for idx, (node, eta) in enumerate(tqdm(intervention.items())):
            # p, p0 = get_transfer(n, E, settings.beta, allowed, states, node, eta)
            row = experiment5_intervention(
                node,
                eta,
                deepcopy(p),
                deepcopy(p0),
                states,
                E,
                settings.beta,
                settings,
                allowed,
                reduction,
            )
            df.append(row)

    # control condition
    # p, p0 = get_transfer(n, E, settings.beta, allowed, state -1, 0)
    df.append(
        intervene(
            -1,
            0,
            deepcopy(p),
            deepcopy(p0),
            states,
            E,
            settings.beta,
            settings,
            allowed,
            reduction,
        )
    )
    return pd.DataFrame(df)


def simulate_reduced(settings, allowed_start, structure=None, e_func=ising):
    # Compute energy and get transer matrix for temperature
    n = len(settings.g)
    if structure is None:
        A = np.asarray(
            nx.adjacency_matrix(
                settings.g,
                weight="weight",
            ).todense()
        )
        states, allowed = gen_states(n)
    else:
        A, states, allowed = structure

    E = e_func(states, A)
    p, p0 = get_transfer(n, E, settings.beta, allowed, states, -1, 0)

    allowed_starting, mags, p_mag = get_allowed_per_mag(states, p0)
    # add equilibrium case
    allowed_starting[-1] = []

    df = []
    # init model and setup
    m = settings.model(p.copy(), p0.copy(), states)
    # print(f"{mag=} {mag_allowed=}")
    m.setup(allowed=allowed_start)
    # can be differently shaped
    # infer shape implicitly
    H = []
    HC = []
    MI = []
    for step in tqdm(range(settings.steps)):
        h, hc, mi = m.update()
        H.append(h)
        HC.append(hc)
        MI.append(mi)
    row = dict(
        mi=np.asarray(MI).copy(),
        h=np.asarray(H).copy(),
        hc=np.asarray(HC).copy(),
        p0=m.p_state.copy(),
    )
    # convert cupy back to numpy
    if hasattr(h, "get"):
        for k in "mi h hc mag p0".split():
            try:
                row[k] = row[k].get()
            except:
                continue

    df.append(row)

    df = pd.DataFrame(df)
    return df


def simulate(settings, structure=None, e_func=ising):
    # Compute energy and get transer matrix for temperature
    n = len(settings.g)
    if structure is None:
        A = np.asarray(
            nx.adjacency_matrix(
                settings.g,
                weight="weight",
            ).todense()
        )
        states, allowed = gen_states(n)
    else:
        A, states, allowed = structure

    E = e_func(states, A)
    p, p0 = get_transfer(n, E, settings.beta, allowed, states, -1, 0)

    allowed_starting, mags, p_mag = get_allowed_per_mag(states, p0)
    # add equilibrium case
    allowed_starting[-1] = []

    df = []
    for mag, mag_allowed in allowed_starting.items():
        # init model and setup
        # test = True
        # if test and mag - 0.05 == 0.5:
        # print("using other p0")
        # p0_ = np.asarray(pd.read_pickle("numerical_p0.pkl"))

        # import proplot as plt
        # fig, ax = plt.subplots()
        # for idx in mag_allowed:
        #     ax.axvline(idx)
        # ax.plot(p0.get())
        # fig.show()
        # plt.show(block=1)
        # print(p0.sum())
        # m = settings.model(p.copy(), p0_.copy(), states)

        # else:
        m = settings.model(p.copy(), p0.copy(), states)

        # print(f"{mag=} {mag_allowed=}")
        m.setup(allowed=mag_allowed)
        # can be differently shaped
        # infer shape implicitly
        H = []
        HC = []
        MI = []
        ps = []
        psc = []
        for step in tqdm(range(settings.steps)):
            h, hc, mi, pi, pj = m.update()
            # h, hc, mi = m.update()
            H.append(h)
            HC.append(hc)
            MI.append(mi)
            ps.append(pi)
            psc.append(pj)
        row = dict(
            mi=np.asarray(MI).copy(),
            h=np.asarray(H).copy(),
            hc=np.asarray(HC).copy(),
            mag=mag,
            p0=m.p_state.copy(),
            pst=np.asarray(ps),
            psc=np.asarray(psc),
        )
        # convert cupy back to numpy
        if hasattr(h, "get"):
            for k in "mi h hc mag p0 pst psc".split():
                try:
                    row[k] = row[k].get()
                except:
                    continue

        df.append(row)
    # convert cupy back to numpy
    if hasattr(mags, "get"):
        mags = mags.get()
        p_mag = p_mag.get()
        p = p.get()
        p0 = p0.get()

    df = pd.DataFrame(df)
    df.attrs = dict(
        mag_bin=mags,
        p_mag=p_mag,
        transfer_matrix=p,
        settings=settings,
        p_state=p0,
    )
    return df


# fig, ax = plt.subplots()
# ax.imshow(states, aspect="auto")
# fig.show()


def store_results(df, name):
    fn = f"./data/exact_{name}.pkl"
    df._name = name
    df = pd.to_pickle(df, fn)


def small_p():
    return nx.LCF_graph(5, [2], 1)
    return nx.path_graph(3)


# compute energy
if __name__ == "__main__":

    T = 30
    beta = 0.5
    # g = double_circle
    # g = small_tree_cross
    g = nx.path_graph(10)
    g.__name__ = "test"
    n = len(g)
    name = g.__name__
    from plexsim.utils.graph import recursive_tree

    # g = recursive_tree(4)
    # g = nx.LCF_graph(n=10, shift_list=[-2], repeats=4)
    g = nx.convert_node_labels_to_integers(g)

    settings = Settings(1.0, T, g, NodeToSystem)
    df = simulate(settings)

    target = 0.2
    I = df[df.mag.round(1) == target].mi.iloc[0]

    bins, p_mag = df.attrs["mag_bin"], df.attrs["p_mag"]
    fig = pplt.figure(sharex=0, sharey=0)
    layout = [[3, 1], [3, 2]]

    # ax = fig.subplot_mosaic(layout)
    ax = fig.add_subplots(layout)

    ax[0].set_title("Transition probability")
    h = ax[0].imshow(df.attrs["transfer_matrix"], aspect="auto", vmin=0, vmax=1)
    ax[0].colorbar(h, loc="r")

    ax[1].set_title(f"State probability steps")
    ax[1].set_xlabel("Magnetization $M(S)$")
    ax[1].set_ylabel("PMF")
    # h = ax[1].imshow(B, aspect="auto", vmin=0, vmax=1)
    # ax[1].colorbar(h, loc="r")

    ax[1].bar(bins, p_mag)

    if target == -1:
        target = "NA"
    else:
        target = round(target, 2)
    ax[2].set_title(
        f"Information decay\n max ent = {n*np.log2(2)} \t distance = {target}"
    )
    ax[2].set_ylim(0, 1.2)
    ax[2].format(xlabel="t", ylabel="$I(s_i^{0} ; S^{\\tau})$")
    from utils import ccolors

    colors = ccolors(n)
    from matplotlib.pyplot import Line2D

    h = []
    for node in range(n):
        lab = f"{node=}, {g.degree(node)=}"
        ax[2].plot(I[:, node], color=colors[node])
        hi = Line2D([], [], color=colors[node], label=lab, marker="o", linestyle="none")
        h.append(hi)
    # ax[2].axhline(0)
    h.append(Line2D([], [], color="tab:blue", label="System"))
    ax[2].legend(handles=h, loc="ul", ncols=2)
    inax = ax[2].inset((0.75, 0.75, 0.25, 0.25), zoom=False)
    nx.draw(g, ax=inax, node_color=colors)
    fig.show()

    plt.show(block=1)
