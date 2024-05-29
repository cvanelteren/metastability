from iFlow.exact import *


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
