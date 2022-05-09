import networkx as nx, numpy as np
from exact import *
from tqdm import tqdm


def get_paths(start, allowed, cutoff):
    paths = {0: ((start,),)}
    print("Constructing paths")
    for t in tqdm(range(1, cutoff + 1)):
        for path in tqdm(paths[t - 1]):
            # we are at this node now
            start = path[-1]
            for other in allowed[start]:
                p = (*path, other)
                paths[t] = paths.get(t, []) + [p]
    return paths[t]


def construct_trajectories(g, settings, e_func=ising):
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
    p, p0 = get_transfer(n, E, settings.beta, allowed)

    # under glauber we have n + 1 possible states we can evolve into
    paths_ = get_paths(0, allowed, cutoff=settings.steps)
    # store the probability of transitioning
    paths = {}
    buffer = np.zeros(settings.steps)
    from copy import deepcopy

    for path in tqdm(paths_):
        buffer.fill(0)
        for idx, (source, target) in enumerate(zip(path[:-1], path[1:])):
            buffer[idx] = p[source, target]
        paths[tuple(path)] = buffer.copy()
    return paths


if __name__ == "__main__":
    g = nx.krackhardt_kite_graph()
    beta = 0.5732374683235916
    # ignore model here, not needed
    settings = Settings(beta=beta, g=g, steps=5, model=None)
    paths = construct_trajectories(g, settings)
    df = []
    for state, ps in paths.items():
        row = dict(state=state, ps=np.array(ps))
        df.append(row)
    df = pd.DataFrame(df)
    df.to_pickle("kite_exact_trajectories.pkl")
