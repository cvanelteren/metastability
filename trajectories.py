import networkx as nx, numpy as np
from functools import partial
from exact import *
from tqdm import tqdm


def get_path(
    path,
    states: np.ndarray,
    p: np.ndarray,
    allowed: dict,
    local_max=False,
):
    """Generate the possible paths from the starting location

    Parameters
    ----------
    path : list of tuples
        contains the starting state from which to do a local
        search
    states : np.ndarray
        state matrix contatining 2**n x n states
    p : np.ndarray
        transfer matrix
    allowed : dict
        allowed state transitions for each state
    local_max : bool
        if true, only keeps paths that maximize the local probability

    Examples
    --------
    FIXME: Add docs.


    """
    # we are at this node now
    paths = []
    start = path[-1]
    ps = []
    for other in allowed[start]:
        # TODO: this is temporary to speedup the trajectories
        if np.mean(states[other]) > np.mean(states[start]):
            pi = (*path, other)
            paths.append(pi)
            ps.append(p[start, other])

    if local_max:
        p_max = np.max(np.array(ps))
        idxs = np.where(ps == p_max)[0]
        return [paths[idx] for idx in idxs]
    return paths


def get_paths(paths, states, p, allowed, cutoff, local_max=False):
    """Multiprocessing wrapper around get_path"""
    import multiprocessing as mp

    print("Constructing paths")

    # paths = [
    #     (start,),
    # ]

    f = partial(
        get_path,
        states=states,
        p=p,
        allowed=allowed,
        local_max=local_max,
    )

    for t in tqdm(range(1, cutoff + 1)):
        new_path = []
        with mp.Pool(mp.cpu_count() - 1) as p:
            for path in p.imap_unordered(f, tqdm(paths)):
                [new_path.append(i) for i in path]
        paths = new_path
        pd.to_pickle(paths, "tmp.pkl")
    return paths


def construct_trajectories(g, settings, e_func=ising, target=0):
    # we start with the system in all zeros
    # after t = 5 steps the system could have reached a tipping point
    # the transfer matrix encodes P(S^t | S^{t-1})
    # we therefore need to know the joint P(S^0, S^1, ..., S^5)
    # to get all possible trajectories.
    #
    # We need to:
    #   - Get the indices of the system that have magnetization M(S) = 0.5
    #   - I have already a dict of allowed transitions
    #

    n = len(g)
    A = nx.adjacency_matrix(g).todense()
    states, allowed = gen_states(n)
    E = e_func(states, A)
    p, p0 = get_transfer(n, E, settings.beta, allowed, states)
    # under glauber we have n + 1 possible states we can evolve into
    # Evolving from the states at the tipping point
    #
    test = False
    if test:
        fp = f"kite_exact_trajectories_local_target=0_steps=5.pkl"
        tmp = pd.read_pickle(fp)
        s = states[np.stack(tmp.state)[:, -1]]
        idxs = np.where(s.mean(1) == target)[0]
        idxs = np.unique(np.stack(tmp.state)[idxs, -1])

        paths = [(idx,) for idx in idxs]

    else:
        paths = [(idx,) for idx, state in enumerate(states) if np.mean(state) == target]

    print(len(paths), target)
    # idxs = [idx for idx, state in enumerate(states) if np.mean(state) == target]
    # tmp = p[np.asarray(idxs)]
    # jdxs = np.where(tmp == np.max(tmp))[0]
    # paths = [(idxs[idx],) for idx in jdxs]

    # paths = []
    # paths = [(303,), (663,)]
    paths_ = get_paths(
        paths,
        states,
        p,
        allowed,
        cutoff=settings.steps,
        local_max=settings.local_max,
    )
    # store the probability of transitioning
    paths = {}
    buffer = np.zeros(settings.steps + 1)
    from copy import deepcopy

    for path in tqdm(paths_):
        buffer.fill(0)
        buffer[0] = p0[path[0]]
        for idx, (source, target) in enumerate(zip(path[:-1], path[1:])):
            buffer[idx + 1] = p[source, target]
        paths[tuple(path)] = buffer.copy()
    return paths


if __name__ == "__main__":
    g = nx.krackhardt_kite_graph()
    beta = 0.5732374683235916
    # ignore model here, not needed
    steps = 10
    settings = Settings(beta=beta, g=g, steps=steps, model=None)
    local_max = True
    settings.local_max = local_max

    target = 0
    paths = construct_trajectories(g, settings, target=target)
    df = []
    for state, ps in paths.items():
        row = dict(state=state, ps=np.array(ps))
        df.append(row)
    df = pd.DataFrame(df)
    fp = f"kite_exact_trajectories_local_{target=}_{steps=}_{local_max=}.pkl"
    print(fp)
    df.to_pickle(fp)
