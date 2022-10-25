import _paths
from neo4j import GraphDatabase
import networkx as nx
# from networkx.algorithms import approximation
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import pandas as pd
import math
# from karateclub import GL2Vec, Graph2Vec, SF, NetLSD, FGSD, GeoScattering
# from karateclub import FeatherGraph, IGE, LDP
import annotate_hoover_scatter as sc_h
import itertools
from network2tikz import plot
from itertools import product
from networkx.readwrite.json_graph import adjacency

import oems


def get_color(n):
    colors = []
    for i in range(n):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    return(colors)


def neo4jReturn(cypherTxt, driver):

    results = driver.session().run(cypherTxt)
    nodes = list(results.graph()._nodes.values())
    rels = list(results.graph()._relationships.values())

    return (nodes, rels)


def get_graph(
    cypherTxt, wTag, w=False, err=[],
    driver=None
):
    # cypherTxt = cypherTxt.format(sTxt, pidMax)
    if not driver:
        driver = GraphDatabase.driver(
            uri="bolt://localhost:3687", auth=("neo4j", "ivory123"))

    # input(cypherTxt)
    # print(driver)
    nodes, rels = neo4jReturn(cypherTxt, driver)
    G = nx.Graph()

    cs, cm = 0, 0
    for node in nodes:
        propN = node._properties
        label, = node._labels
        if label == 'Sim':
            cs += 1
            L = propN['sim_abb']
            if L == int:
                L = '0' + str(int(L) - 3)
            G.add_node(node.id, name=L, properties=propN,
                       bipartite=1, label='Sim')
    # for node in nodes:
    #     propN = node._properties
    #     label, = node._labels
        if label == 'Des':
            cm += 1
            L = propN['des_pid']
            G.add_node(node.id, name=L, properties=propN,
                       bipartite=0, label='Des')
    # for node in nodes:
    #     propN = node._properties
    #     label, = node._labels
    #     if label == 'Behav':
    #         L = propN['behav_id']
    #         G.add_node(node.id, name=L, properties=propN, bipartite=1, label='Behav')

    # print('Sim: ', cs)
    # print('Des: ', cm)
    for rel in rels:
        if not rel.start_node in G or not rel.end_node in G:
            'works'

        src = rel.start_node.id
        trgt = rel.end_node.id
        G.add_edge(
            src,
            trgt,
            color='k',
            key=rel.id, type=rel.type, properties=rel._properties)

        if w:
            sampled_edge = (src, trgt)
            # input(rel._properties['weight'])
            w_list = rel._properties['w_e_value']
            wi = rel._properties['w_e_key'].index(wTag)
            weight = w_list[wi] / w

            # input(weight)
            if math.isinf(weight):
                weight = 0.0001

            # print(G.nodes()[rel.start_node.id]['name'], weight)
            nx.set_edge_attributes(
                G, {sampled_edge: {'weight': weight}})
        # else:
            # print('no')
    start_index = 0
    G = nx.convert_node_labels_to_integers(G, first_label=start_index)

    return(G)


def get_graph_old(cypherTxt, nc, sTxt=''):
    results = driver.session().run(cypherTxt.format(sTxt))
    G = nx.Graph()

    # subset_sizes = [4, 4]
    # subset_sizes = [5, 5, 4, 3, 2, 4, 4, 3]
    # extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    # layers = [range(start, end) for start, end in extents]
    # print(layers)

    nodes = list(results.graph()._nodes.values())
    colors = []
    for node in nodes:
        label, = node._labels
        L = label
        if label == 'Des':
            L = node._properties['des_pid']
        if label == 'Sim':
            L = node._properties['sim_abb']
        G.add_node(node.id, labels=L, properties=node._properties)
        colors.append(nc[label])

    rels = list(results.graph()._relationships.values())
    for rel in rels:
        G.add_edge(
            rel.start_node.id,
            rel.end_node.id,
            color='k',
            # weight=rel._properties['weight'],
            key=rel.id, type=rel.type, properties=rel._properties)
    print(type(rel._properties['weight']))
    # changing the indices - needed for karateclub algorithms
    # (continuous indexing)
    start_index = 0
    G = nx.convert_node_labels_to_integers(G, first_label=start_index)

    seed = 0
    for li in G.nodes.data('labels'):
        if li[1] == 'Sim':
            seed = li[0]

    # labels = nx.get_node_attributes(G, 'labels')
    # options = {"node_color": colors, "node_size": 100, "labels": labels}
    # nx.draw_spectral(G, **options)
    # nx.draw(G, **options)
    # plt.show()
    return(G, colors, seed)


def graph_sims_str(nSim, cypherTxt, nc):
    cypherTxtSim = """
        match (s:Sim) return s limit {}
    """.format(nSim)
    results = driver.session().run(cypherTxtSim)
    nodes = list(results.graph()._nodes.values())

    GS = []
    colors = []
    sNames = []
    seeds = []
    for n in nodes:
        simName = n._properties["sim_name"]
        sTxt = "{{sim_name:'{0}'}}".format(simName)
        Gi, ci, si = get_graph(cypherTxt, nc, sTxt)

        # nx.write_edgelist(Gi, "{}.txt".format(simName), data=False)
        GS.append(Gi)
        colors.append(ci)
        sNames.append(n._properties["sim_abb"])
        seeds.append(si)
    if len(set(seeds)) == 1:
        s = seeds[0]
    else:
        print(seeds)
        s = 42
    return(GS, colors, sNames, s)


def plot_all_models(GS, colors, sNames, seed):

    gEmbdModel = [
        'GL2Vec', 'Graph2Vec', 'SF', 'NetLSD', 'FGSD',
        'GeoScattering', 'FeatherGraph', 'IGE', 'LDP']
    colors = get_color(len(sNames))

    fig, axs = plt.subplots(3, 3)
    for i, m in enumerate(gEmbdModel):
        print(m)
        # model = Graph2Vec(dimensions=2, wl_iterations=100, epochs=2)GraphWave
        model = eval(m + "()")
        model.__dict__ = eval('gModels.{}.dic'.format(m))
        model.seed = seed
        print(model.__dict__)

        model.fit(GS)
        embds = model.get_embedding()
        x = list(embds[:, 0])
        y = list(embds[:, 1])

        r = i % 3
        c = int(i / 3)
        fig, ax = sc_h.scatter_hoover(
            x, y, sNames, colors, fig, axs[r, c],
            'sim_abb', leg=None)

        # break
        ax.set_title(m)
    fig.suptitle('Sim-Des-Part-Behav', fontsize=16)
    fig.set_size_inches(18.5, 10.5)
    plt.show()


def plot_hyper_prm(GS, colors, sNames, seed):
    gEmbdModel = [
        'GL2Vec', 'Graph2Vec', 'SF', 'NetLSD', 'FGSD',
        'GeoScattering', 'FeatherGraph', 'IGE', 'LDP']
    m = gEmbdModel[5]

    fig, axs = plt.subplots(2, 5)
    colors = get_color(len(sNames))

    for i, v in enumerate(range(5, 51, 5)):
        print(v)
        # model = Graph2Vec(dimensions=2, wl_iterations=100, epochs=2)GraphWave
        model = eval(m + "()")
        model.__dict__ = eval('gModels.{}.dic'.format(m))
        model.seed = v
        # model.moments = v
        # model.order = v
        # print(model.__dict__)

        model.fit(GS[:v])
        embds = model.get_embedding()
        x = list(embds[:, 0])
        y = list(embds[:, 1])

        c = i % 5
        r = int(i / 5)
        fig, ax = sc_h.scatter_hoover(
            x, y, sNames[:v], colors[:v], fig, axs[r, c],
            'sim_abb', leg=None)

        ax.set_title('{0}- NO. input graph:{1}'.format(m, v))
    fig.suptitle('Sim-Des-Part-Behav', fontsize=16)
    fig.set_size_inches(18.5, 10.5)
    plt.show()

    return(G)


def w_cn(G, ebunch=None, alpha=0.8):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    shortest_path = nx.shortest_path(G, weight='weight', method='dijkstra')
    shortest_path1 = nx.shortest_path(G, weight=None)
    # spaths = dict(nx.all_pairs_dijkstra_path(G, weight=weight))

    print([G[u][v]['weight'] for u, v in G.edges()])
    for u, v in nx.edges(G):
        print(G.get_edge_data(u, v)['weight'])

    vals = []
    for u, v in ebunch:
        # resource_allocation_index
        # print(u, v, sum(1 / G.degree(w) for w in nx.common_neighbors(G, u, v)))
        # print(G)
        # print(len(list(nx.common_neighbors(G, u, v))))
        # print(shortest_path1[u][v])
        # print(shortest_path[u][v])
        val = alpha * len(list(nx.common_neighbors(G, u, v))) + (1 - alpha) * (
            G.number_of_nodes() / (len(shortest_path[u][v]) - 1)
        )
        vals.append([u, v, val])

        print(f"({u}, {v}) -> {val:.8f}")
        print(G.number_of_edges(), G.number_of_nodes())

    print('--------------------')
    return(vals)


def simRank(pidMax, sLimit):
    style = cyTxt()
    G = get_graph(
        cyTxt.sm.txt, style.nodeColor, pidMax=pidMax)
    top, btw = nx.bipartite.sets(G)
    pos = nx.bipartite_layout(G, top)
    '''
    not weighted graph shoulb loaded to use the numpy version otherwise use
    numpy adjance is puting w instead of 1.

    sim = nx.simrank_similarity(G, max_iterations=1000)
    lol = np.array([[sim[u][v] for v in sorted(sim[u])] for u in sorted(sim)])
    '''
    sim = nx.simrank_similarity_numpy(G, max_iterations=1000000)
    lol = sim

    # ---------
    # Bipartite
    # ---------
    style.node_label_pos['Des'] = 'below'
    # plot(G, **style.style(G, pos))
    sNode = [
        k for k, v in nx.get_node_attributes(G, 'label').items() if v == 'Sim']
    for i, row in enumerate(lol):
        for j, s in enumerate(row):
            if not i == j:
                if s > sLimit:
                    if i in sNode and j in sNode:
                        G.add_edge(i, j, weight=s, type='SIM_SIM')

    layout2 = nx.kamada_kawai_layout(G)
    style.node_label_pos['Des'] = '180'
    styleAdd = {
        'canvas': (18, 18),
        'edge_curved': 0.2,
        'edge_label_distance': 0.3
    }
    # plot(G, **style.style(G, layout2, w='weight'), **styleAdd)
    G2 = subGraph_sim(G)
    edge_w = print_sort_weight(G2)
    return(edge_w)


def add_simsim(G, simMatrix, sLimit):
    sNode = [
        k for k, v in nx.get_node_attributes(G, 'label').items() if v == 'Sim']

    # for e in G.edges():
    #     G.remove_edge(e[0], e[1])

    for i, row in enumerate(simMatrix):
        for j, s in enumerate(row):
            if not i == j:
                if s > sLimit:
                    if i in sNode and j in sNode:
                        G.add_edge(i, j, weight=s, type='SIM_SIM')
    # remove disconnected sims
    # G.remove_nodes_from(list(nx.isolates(G)))
    return(G)


def simRankpp(G, pidMax, sLimit, wscl=1, evd=True, sprd=True, importance_factor=0.9):
    style = cyTxt()
    # nx.draw(G, with_labels=True)
    # plt.show()
    # input('wait')
    try:
        # print(pidMax)
        top, btw = nx.bipartite.sets(G)
    except:
        print('ERROR NOT BIPARTITE GRAPH')
        # print(G.nodes(data=True)[231])
        nx.draw(G, with_labels=True)
        plt.show()
        return(G, None, None, None)
    pos = nx.bipartite_layout(G, top)

    labels = {}
    for i, n in G.nodes(data=True):
        labels[i] = (n['name'])
    ws = nx.get_edge_attributes(G, 'weight')

    '''
    not weighted graph shoulb loaded to use the numpy version otherwise use
    numpy adjance is puting w instead of 1.

    sim = nx.simrank_similarity(G, max_iterations=1000)
    lol = np.array([[sim[u][v] for v in sorted(sim[u])] for u in sorted(sim)])
    '''
    # sim = nx.simrank_similarity_numpy(G, max_iterations=1000000)
    sNode = [
        k for k, v in nx.get_node_attributes(G, 'label').items() if v == 'Sim']
    sim = simrank_pp_similarity_numpy(
        G, max_iterations=1000000, evd_opt=evd, sprd_opt=sprd, importance_factor=importance_factor)  # , source=sNode)
    lol = sim
    # sim = sim[sNode, :]
    # ---------
    # Bipartite
    # ---------
    style.node_label_pos['Des'] = 'below'
    # plot(G, **style.style(G, pos))
    # print(sim)

    # -------------
    # add sim-sim
    # -------------
    # simsim = []
    # for i, row in enumerate(lol):
    #     for j, s in enumerate(row):
    #         if not i == j:
    #             if s > sLimit:
    #                 # print(i, j)
    #                 if i in sNode and j in sNode:
    #                     G.add_edge(i, j, weight=s, type='SIM_SIM')
    #                     simsim.append([i, j])
    simsim = []
    for ni in sNode:
        for nj in sNode:
            if not ni == nj and nj > ni:
                s = sim[ni][nj]
                if s > sLimit:
                    G.add_edge(ni, nj, weight=s, type='SIM_SIM')
                    simsim.append([ni, nj])
    # print(sim[sNode, :][:, sNode])
    # print(len(sNode))
    # print(sim[sNode, sNode].shape)
    # print('------------------')

    # G2 = plot_subgraph(G, pidMax, wscl, style)

    # sort edge based on weights
    # edge_w = print_sort_weight(G)
    return(G, simsim, sim, sNode)


def subGraph_sim(G, name=''):

    sims = [v[0] for v in G.nodes(data=True) if v[1]['label'] == 'Sim']
    G.add_node(1000, label='picName', name=name)
    G2 = nx.subgraph(G, sims + [1000])

    return(G2)


def plot_subgraph(B, pidMax, wscl, style):
    name = 'part_{}_wscl_{}_spread_off'.format(pidMax, wscl)
    outName = 'pic/link_pred_{}.pdf'.format(name)
    # G.add_node(1000, label='picName', name=

    G2 = subGraph_sim(B, name=name.replace('_', '-'))
    layout2 = nx.shell_layout(G2)
    style.node_label_pos['Des'] = '180'
    styleAdd = {
        'canvas': (8, 8),
        'edge_curved': 0.2,
        'edge_label_distance': 0.3
    }
    # plot(G2, **style.style(G2, layout2, w='weight'), **styleAdd)
    # plot(G2, outName, **style.style(G2, layout2, w='weight'), **styleAdd)
    '''
    find . -type f -name '*.pdf' -print0 |
    while IFS= read -r -d '' file
        do convert -verbose -density 500 -resize 800 "${file}" "${file%.*}.png"
        done

        '''
    return(G2)


def print_sort_weight(B, nPred=6):
    ''' wrong function, it doesn't get the sim-sim edges, it gets all the edges'''
    edge_w = []
    edges = sorted(
        B.edges(data=True), key=lambda t: t[2].get('weight', 1),
        reverse=True)
    nodes = B.nodes(data=True)
    if len(edges) == 0:
        edge_w = ['-' for e in range(0, nPred)]

    for e in edges:
        u, v = e[0], e[1]
        # out = (str(nodes[u]['name']) + '-' + str(nodes[v]['name']) +
        # ', {:.3f};'.format(B[u][v]['weight']))
        out = B[u][v]['weight']
        # print(out)

        # print(out)
        edge_w.append(out)

    return(edge_w)


def evidence(G, opt):
    '''
    chose among two differnt method

    opt=1 , evidence(a, b) = 1 − e^-(|E(a) common E(b)| )
    opt=2 , evidence(a, b) = sum(1 /2^i)
        '''

    nNodes = G.number_of_nodes()
    evdi = np.zeros((nNodes, nNodes))
    nodes = np.array(list(G.nodes()))
    evd2 = np.zeros((nNodes, nNodes))

    for u in G.nodes():
        for v in G.nodes():
            cn_nbr = sorted(nx.common_neighbors(G, u, v))
            uu = np.where(nodes == u)[0][0]
            vv = np.where(nodes == v)[0][0]
            s = 0
            for i in range(1, len(cn_nbr) + 1):
                s += 1 / pow(2, i)
            evd2[uu, vv] = s
            evdi[uu, vv] = len(cn_nbr)
    if opt == 1:
        evd = 1 - np.exp(-evdi)
        np.fill_diagonal(evd, 1.0)
        return(evd)
    if opt == 2:
        np.fill_diagonal(evd2, 1.0)
        return(evd2)


def spread(G, opt, old=False):

    adjacency_matrix = nx.to_numpy_array(G)

    if old:
        normalized_w = adjacency_matrix / adjacency_matrix.sum(axis=1)
        var = np.nanvar(
            np.where(adjacency_matrix == 0, np.nan, adjacency_matrix), axis=0)

    # vaiance for normalized weights /
        var = np.nanvar(
            np.where(adjacency_matrix == 0, np.nan, normalized_w), axis=0)
        spread = np.exp(-var)
        normalized_w = adjacency_matrix / adjacency_matrix.max(axis=1)
        W = spread * normalized_w
        return (W)

    nNodes = G.number_of_nodes()
    sprd = np.zeros((nNodes, nNodes))

    W = np.zeros((nNodes, nNodes))
    var = np.nanvar(
        np.where(adjacency_matrix == 0, np.nan, adjacency_matrix), axis=0)
    sprd = np.exp(-var)

    for i, r in enumerate(adjacency_matrix):
        normalized_w = r / r.sum()

        W[i, :] = sprd * normalized_w

    W = np.zeros((nNodes, nNodes))
    for r, row in enumerate(W):
        for c, cel in enumerate(row):
            if opt == 'src':
                W[r, c] = (sprd[c] * adjacency_matrix[r, c] /  # normalized vs source
                           adjacency_matrix[r, :].sum())
            elif opt == 'trgt':
                W[r, c] = (sprd[c] * adjacency_matrix[r, c] /  # normalized vs target
                           adjacency_matrix[:, c].sum())

    return(W)


def w_func(G):
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    return(weights)


def simrank_pp_similarity_numpy(
    G,
    source=None,
    target=None,
    importance_factor=0.9,
    max_iterations=100,
    tolerance=1e-5,
    evd_opt=2,
    sprd_opt=False,


):
    adjacency_matrix = nx.to_numpy_array(G)

    if sprd_opt:
        adjacency_matrix = spread(G, sprd_opt)
    else:
        adjacency_matrix /= adjacency_matrix.sum(axis=0)
    if evd_opt:
        evd = evidence(G, evd_opt)

    newsim = np.eye(adjacency_matrix.shape[0], dtype=np.float64)
    # newsim = np.multiply(evd, newsim)
    for i in range(max_iterations):

        prevsim = np.copy(newsim)
        newsim = importance_factor * np.matmul(
            np.matmul(adjacency_matrix.T, prevsim), adjacency_matrix
        )

        if evd_opt:
            newsim = evd * newsim
        np.fill_diagonal(newsim, 1.0)

        if np.allclose(prevsim, newsim, atol=tolerance):
            break

    if source is not None and target is not None:
        return newsim[source, target]
    if source is not None:
        # print(source)
        return newsim[source]
    return newsim


def compare_result(nPart, wscl=10e6):
    data = {}

    # data['simrank nW {}'.format(nPart)] = simRank(nPart, 0.0)
    # data['simrank++ nW, evd {}'.format(nPart)] = simRankpp(
    #     nPart, 0.0, wscl=False, sprd=False)
    # data['simrank++ W-nrm, evd, {}'.format(nPart)] = simRankpp(
    #     nPart, 0.0, wscl=1, sprd=False)
    # data['simrank++ W-nrm, evd, sprd {}'.format(nPart)] = simRankpp(
    #     nPart, 0.0, wscl=1, sprd=True)
    data['simrank Wscl {}, evd {}'.format(str(wscl), nPart)] = simRankpp(
        nPart, 0.0, wscl=wscl, sprd=True)

    df = pd.DataFrame(data)
    dataf = df['simrank Wscl {}, evd {}'.format(str(wscl), nPart)]
    print(max(dataf) - min(dataf))
    # print(dataf)

    return(df)


class simrank_inv:

    def __init__(self, simMatrix, G, simNode, simPair):
        self.simM = simMatrix
        self.G = G
        self.simN = simNode
        self.simP = simPair

    def density(self):
        d = {'score': [], 'sims': []}

        for s in self.simP:
            u, v = s
            w = self.G[u][v]['weight']
            uName = self.G.nodes(data=True)[u]['properties']['sim_name']
            vName = self.G.nodes(data=True)[v]['properties']['sim_name']
            uv = uName + ',' + vName

            d['score'].append(w)
            d['sims'].append(uv)
        return(d)

    def upper_range(self, smin):
        print(np.max(self.simM[self.simN, :][:, self.simN]))

        wList = []
        for s in self.simP:
            u, v = s
            w = self.G[u][v]['weight']
            wList.append(w)

            if w >= smin:
                # print('-------------------')
                rowu = self.simM[u, :]
                rowv = self.simM[v, :]

                valu = np.min(rowu[np.where(rowu > 0)])
                valv = np.amin(rowv[np.where(rowv > 0)])

                iu = np.where(rowu == valu)[0][0]
                iv = np.where(rowv == valv)[0][0]

                print(self.G.nodes(data=True)[u]['properties']['sim_name'])
                print(self.G.nodes(data=True)[v]['properties']['sim_name'])
                print(iu, self.G.nodes(data=True)[
                      iu]['properties']['sim_name'], valu)
                print(iv, self.G.nodes(data=True)[
                      iv]['properties']['sim_name'], valv)

                print(w)
                print('-------------------')
        print(max(wList))

    def get_sim(self, simList):
        nodesG = self.G.nodes(data=True)

        print('----------------------')
        sName = [u[1]['name'] for u in nodesG]
        sId = [u[0] for u in nodesG]
        ids = []
        for s in simList:
            i = sName.index(s)
            rows00 = self.simM[:, i]
            si = self.simM[i, i]

            # find the highest similarity
            rows0 = self.simM[self.simN, i]
            rMaxi = np.max(rows0)
            # ignore similarity to itself
            if rMaxi == si:
                rows = rows0
                rows.sort()
                rMaxi = rows[-2]
            mi = np.nonzero(rows00 == rMaxi)[0][0]
            ids.append([i, mi])

        for pair in ids:
            print(pair)
            print(self.simM[pair[0], pair[1]])
            print('----------------------')

    def simRank_rng(self, nPID, d, opt='max'):

        def set_data(u, v, opt):

            rowu = self.simM[u, :]
            rowv = self.simM[v, :]

            if opt == 'max':
                valu = np.min(rowu[np.where(rowu > 0)])
                valv = np.amin(rowv[np.where(rowv > 0)])
            else:
                valu = np.max(rowu[np.where(rowu < 1)])
                valv = np.amax(rowv[np.where(rowv < 1)])

            iu = np.where(rowu == valu)[0][0]
            iv = np.where(rowv == valv)[0][0]

            d['nPID'].append(nPID)
            d['H1'].append(nodes[u]['name'])  # .split('_')[0])
            d['H2'].append(nodes[v]['name'])  # .split('_')[0])
            d['L1'].append(nodes[iu]['name'])  # .split('_')[0])
            d['L2'].append(nodes[iv]['name'])  # .split('_')[0])
            # d['H1'].append(nodes[u]['properties']['sim_name'])
            # d['H2'].append(nodes[v]['properties']['sim_name'])
            # d['L1'].append(nodes[iu]['properties']['sim_name'])
            # d['L2'].append(nodes[iv]['properties']['sim_name'])
            d['w:H1H2'].append(w)
            d['w:H1L1'].append(valu)
            d['w:H2L2'].append(valv)

        nodes = self.G.nodes(data=True)
        wList = []
        for s in self.simP:
            u, v = s
            w = self.G[u][v]['weight']
            wList.append(w)

        if opt == 'max':
            smax = max(wList)
        else:
            smin = min(wList)

        for s in self.simP:
            u, v = s
            w = self.G[u][v]['weight']

            if opt == 'max':
                if w >= smax:
                    set_data(u, v, opt)
            else:
                if w <= smin:
                    set_data(u, v, opt)
        return(d)


class cyTxt:

    def __init__(self):

        self.nodeColor = {
            'Part': 'red',
            'Behav': 'green',
            'Des': 'blue',  # slate
            'Sim': 'cyan',
            'picName': 'white',
            'Model': 'darkviolet',
            'Barr': 'lightgrey',
            'Veh': '#20B2AA',  # 'lightseegreen',
            'Pltf': '#808000',  # 'olive',
            'Ubdy': 'cornflowerblue'
        }

        self.node_label_pos = {
            'Sim': 'center',
            'Des': 'center',
            'Behav': 'center',
            'Part': 'center',
            'picName': 'below',
            'Model': 'center',
            'Barr': 'center',
            'Veh': 'center',
            'Pltf': 'center',
            'Ubdy': 'center'}

        self.edge_color = {
            'SIM_DES': 'gray',
            'SIM_SIM': 'red',
            'DES_BEHAV': 'gray',
            'NRG_PART': 'gray',
            'PART_DES': 'gray',
            'SIM_BEHAV': 'gray',
            'PART_BEHAV': 'gray',
            'INCL_PART': 'gray',
            'DES_SYM': 'gray',
            'MODEL_REF': 'gray',
            'CNCT_TO': 'gray',
            'MODEL_VEH': 'gray',
            'SIM_MODEL': 'gray',
            'VEH_PLTF': 'gray',
            'VEH_UBDY': 'gray'
        }

        self.nodeSize = {
            'Part': 0.8,
            'Behav': 0.8,
            'Des': 1.1,
            'Sim': 0.8,
            'picName': 0.8,
            'Model': 0.8,
            'Barr': 0.8,
            'Veh': 0.8,
            'Pltf': 0.8,
            'Ubdy': 0.8
        }

    def style(self, G, pos, w=False):
        nodesD = G.nodes(data=True)

        node_color = [self.nodeColor[u[1]['label']] for u in nodesD]
        node_size = [self.nodeSize[u[1]['label']] for u in nodesD]
        node_name = [u[1]['name'] for u in nodesD]
        node_label_pos = [self.node_label_pos[u[1]['label']] for u in nodesD]
        edge_color = [self.edge_color[G[u][v]['type']] for u, v in G.edges()]

        if w:
            # try:
            key = w  # 'weight'
            if key == 'weight':
                edge_label = [
                    '{:.2f}'.format(G[u][v][key]) if key in G[u][v] else ''
                    # '{:d}'.format(int(G[u][v][key])) if key in G[u][v] else ''
                    for u, v in G.edges()]
                # node_pairs= [
                #     '{0}-{1}:'.format(nodesD[u]['name'],nodesD[v]['name']) for u, v in G.edges()]
                # for ei, e in enumerate(edge_label):
                #     print(node_pairs[ei], e)
            else:
                edge_label = [
                    (G[u][v][key]).replace('_', '\_')
                    for u, v in G.edges()]

            # except KeyError:
            # #     edge_label = ['{:.2f}'.format(G[u][v]['weight']) for u, v in G.edges() if]
            #     edge_label = ''
        else:
            edge_label = ''

        visual_style = {
            "vertex_color": node_color,
            "vertex_opacity": .5,
            "vertex_label":  node_name,
            "vertex_label_position": node_label_pos,
            "edge_width": 0.1,
            "layout": pos,
            "margin": 0.8,
            "edge_color": edge_color,
            "edge_label": edge_label,
            "node_size": node_size,
            # "edge_label": ['test', 'test','test','test\_tst']

        }
        return visual_style

    # nodeColor = {
    #     'Part': 'r',
    #     'Behav': 'g',
    #     'Des': 'b',
    #     'Sim': 'c'}

    class schema:
        txt = """
        //view_schema
        call db.schema.visualization
        """

    class spem:
        txt = """
            //embd_nrg_graph
            match (s:Sim{})
            with s limit 30
            match (s)-[rs:NRG_PART]-(p:Part)-[re:PART_BEHAV]-(e:Behav)
            with s,e,p,rs,re
            match (p)-[rm:PART_DES]-(m:Des)
            return s,e,p,m,rm,re,rs
            """

    class spem_name:
        txt = """
            //embd_nrg_graph
            match (s:Sim)
            where s.sim_name=~ '{}'
            match (s)-[rs:NRG_PART]-(p:Part)-[re:PART_BEHAV]-(e:Behav)
            with s,e,p,rs,re
            match (p)-[rm:PART_DES]-(m:Des)
            return s,e,p,m,rm,re,rs
            """

    class sp:
        txt = """
            //embd_nrg_graph
            match (s:Sim{})
            with s limit 30
            match (s)-[rs:NRG_PART]-(p:Part)
            return s,p,rs
            """

        nodeColor = {
            'Part': 'r',
            'Sim': 'c'}

    class spm:
        txt = """
            //embd_nrg_graph
            match (s:Sim{})
            with s limit 30
            match (s)-[rs:NRG_PART]-(p:Part)-[rm:PART_DES]-(m:Des)
            return s,p,m,rs,rm
            """

        nodeColor = {
            'Part': 'r',
            'Des': 'b',
            'Sim': 'c'}

    class sm:
        txt = """
            //embd_nrg_graph
            match (s:Sim{})
            with s limit 30
            CALL{{
                with s
                match p=(s)-[rm:SIM_DES]-(m:Des)
                return m,rm order by rm.w_e_value  desc limit {}
            }}
            return s,m, rm
            """
        # where m.des_type='pid'

        txt_list = """
            //embd_nrg_graph
            match (s:Sim)
            where s.sim_abb in {}
            with s limit 30
            CALL{{
                with s
                match p=(s)-[rm:SIM_DES]->(m:Des)
                return m,rm order by rm.w_e_value  desc  limit {}
            }}
            return s,m, rm order by s.sim_name
            """

        nodeColor = {
            'Des': 'blue',
            'Sim': 'cyan'}

    class sm_err:
        txt = """
            //embd_nrg_graph
            match (s:Sim{})
            where not s.sim_name in ["{}"]
            with s limit 30
            CALL{{
                with s
                match p=(s)-[rm:SIM_DES]-(m:Des)
                return m,rm order by rm.w_e_value  desc limit {}
            }}
            return s,m, rm
            """
        # where m.des_type='pid'

        nodeColor = {
            'Des': 'b',
            'Sim': 'c'}

    class sm_name:
        txt = """
            //embd_nrg_graph
            match (s:Sim)
            where s.sim_name=~ '{}'
            CALL{{
                with s
                match p=(s)-[rm:SIM_DES]-(m:Des)
                return m,rm order by rm.w_e_value desc limit {}
            }}
            return s,m, rm order by s.sim_name
            """
        # return m,rm order by rm.w_e_value  desc limit {}

        txt_list = """
            //embd_nrg_graph
            match (s:Sim)
            where s.sim_abb in {}
            CALL{{
                with s
                match p=(s)-[rm:SIM_DES]-(m:Des)
                return m,rm order by rm.w_e_value desc limit {}
            }}
            return s,m, rm order by s.sim_name
            """

    class sm_name_err:
        txt = """
            //embd_nrg_graph
            match (s:Sim)
            where s.sim_name=~ '{}' and not s.sim_name in ["{}"]
            CALL{{
                with s
                match p=(s)-[rm:SIM_DES]-(m:Des)
                return m,rm order by rm.w_e_value  desc limit {}
            }}
            return s,m, rm order by s.sim_name
            """
        # return m,rm order by rm.w_e_value  desc limit {}


class gModels:

    class mGL2Vec:
        dic = {
            'wl_iterations': 2, 'dimensions': 2, 'workers': 4,
            'down_sampling': 0.0001, 'epochs': 10, 'learning_rate': 0.025,
            'min_count': 5, 'seed': 42, 'erase_base_features': False}

        dic_defualt = {
            'wl_iterations': 2, 'dimensions': 128, 'workers': 4,
            'down_sampling': 0.0001, 'epochs': 10, 'learning_rate': 0.025,
            'min_count': 5, 'seed': 42, 'erase_base_features': False}

    class mGraph2Vec:
        dic = {
            'wl_iterations': 2, 'attributed': False, 'dimensions': 2,
            'workers': 4, 'down_sampling': 0.0001, 'epochs': 10,
            'learning_rate': 0.025, 'min_count': 5, 'seed': 42,
            'erase_base_features': False}

        dic_defualt = {
            'wl_iterations': 2, 'attributed': False, 'dimensions': 128,
            'workers': 4, 'down_sampling': 0.0001, 'epochs': 10,
            'learning_rate': 0.025, 'min_count': 5, 'seed': 42,
            'erase_base_features': False}

    class mSF:
        dic = {
            'dimensions': 2, 'seed': 42}

        dic_defualt = {
            'dimensions': 128, 'seed': 42}

    class mNetLSD:
        dic = {
            'scale_min': -2.0, 'scale_max': 2.0, 'scale_steps': 250,
            'approximations': 200, 'seed': 42}

        dic_defualt = {
            'scale_min': -2.0, 'scale_max': 2.0, 'scale_steps': 250,
            'approximations': 200, 'seed': 42}

    class mFGSD:
        dic = {
            'hist_bins': 200, 'hist_range': (0, 20), 'seed': 42}

        dic_defualt = {
            'hist_bins': 200, 'hist_range': (0, 20), 'seed': 42}

    class mGeoScattering:
        dic = {
            'order': 4, 'moments': 4, 'seed': 42}

        dic_defualt = {
            'order': 4, 'moments': 4, 'seed': 42}

    class mFeatherGraph:
        dic = {
            'order': 5, 'eval_points': 25, 'theta_max': 2.5, 'seed': 42,
            'pooling': 'mean'}

        dic_defualt = {
            'order': 5, 'eval_points': 25, 'theta_max': 2.5, 'seed': 42,
            'pooling': 'mean'}

    class mIGE:
        dic = {
            'feature_embedding_dimensions': [3, 5],
            'spectral_embedding_dimensions': [10, 20],
            'histogram_bins': [10, 20], 'seed': 42}

        dic_defualt = {
            'feature_embedding_dimensions': [3, 5],
            'spectral_embedding_dimensions': [10, 20],
            'histogram_bins': [10, 20], 'seed': 42}

    class mLDP:
        dic = {
            'bins': 32}

        dic_defualt = {
            'bins': 32}

# if __name__ == '__main__':

    # simRank ++
    # ----------------------------------------------------------------------------

    # for w in [1]:#, 10e6]:
    #
    #     simRankpp(4, 0.0, wscl=w)
        # simRankpp(8, 0.0, wscl=w)
        # simRankpp(12, 0.0, wscl=w)
        # simRankpp(20, 0.0, wscl=w)
        # simRankpp(28, 0.0, wscl=w)

    # fPath = '/home/apakiman/Nextcloud/PhD/03_project/sim.xlsx'
    # with pd.ExcelWriter(fPath) as f:
    #     for p in [2, 5, 15, 28]:  # [4, 5, 8, 12, 20, 28]:  # [5]
    #         df = compare_result(p, wscl=10e8)
    #         df.to_excel(f, sheet_name='part_{}'.format(p))

    # fPath = '/home/apakiman/Nextcloud/PhD/03_project/sim_2.xlsx'
    # with pd.ExcelWriter(fPath) as f:
    #     data = {}
    #     for p in [4, 8, 12, 20, 28]:
    #         data['simrank nW {}'.format(nPart)] = simRank(nPart, 0.0)

        # df = pd.DataFrame(data)
        # df.to_excel(f, sheet_name='part_{}'.format('simrank nW'))
    # -----------------------------------------------
    # ----------------------------------------------------------------------------
    #  weisfeiler_lehman_graph_hash
    # ----------------------------------------------------------------------------
    # pidMax = 4
    # style = cyTxt()
    # G, c, _ = get_graph_bipartie(style.sm.txt, style.nodeColor2, pidMax=pidMax)
    # top, btw = nx.bipartite.sets(G)
    # pos = nx.bipartite_layout(G, top)
    # style = cyTxt()
    # # plot(G)
    # a = nx.weisfeiler_lehman_graph_hash(G)
    # print(len(a))
    # print(G.nodes(data=True)[0])
    # layout2 = nx.kamada_kawai_layout(G)
    # style.node_label_pos['Des'] = '180'
    # styleAdd = {
    #     'canvas': (18, 18),
    #     'edge_curved': 0.2,
    #     'edge_label_distance': 0.3
    # }
    # plot(
    #     G, **style.style(G, layout2, w=True), **styleAdd)

    # ----------------------------------------------------------------------------
    # graph embedding loop for different methodsp
    # ----------------------------------------------------------------------------
    # GS, colors, sNames, seed = graph_sims_str(60, cyTxt.spem.txt, cyTxt.spem.nodeColor)
    # GS, colors, sNames, seed = graph_sims_str(60, cyTxt.sp.txt, cyTxt.spem.nodeColor)
    # plot_all_models(GS, colors, sNames, seed)
    # plot_hyper_prm(GS, colors, sNames, seed)
    # print(GS)
    # ----------------------------------------------------------------------------
    # Link prediction
    # ----------------------------------------------------------------------------
    # G, c, _ = get_graph(cyTxt.sm.txt, cyTxt.sm.nodeColor)
    #
    # # preds = nx.jaccard_coefficient(G)
    # epochs = [(0, 5), (0, 7), (0, 8), (5, 7), (5, 8), (7, 8)]
    # # epochs = [(0,6), (1,5),(1,6)]
    # # preds = nx.resource_allocation_index(G, epochs)#, [(0, 21), (0, 24), (0, 25), (21, 24), (21, 25), (24, 25)])
    # preds = w_cn(G)#, epochs)
    # # preds = nx.common_neighbor_centrality(G, epochs)#, [(0, 21), (0, 24), (0, 25), (21, 24), (21, 25), (24, 25)])
    # #preds = nx.common_neighbor_centrality(G, epochs)#, [(0, 21), (0, 24), (0, 25), (21, 24), (21, 25), (24, 25)])
    #
    #
    # print(preds)
    # for u, v, p in preds:
    #     print(p)
    #     if p  > 4:
    #         G.add_edge(u, v, weight=p)
    #
    # labels = nx.get_node_attributes(G, 'labels')
    # # ec = [G[u][v]['color'] for u,v in G.edges()]
    # w = [G[u][v]['weight'] for u,v in G.edges()]
    #
    # W = []
    # # for wi in w:
    # #     if wi > 2: W.append(wi/10e6)
    #     #else: W.append(wi)
    # options = {
    #     "node_color": c, "node_size": 100,
    #     "labels": labels, 'edge_color':w}#, 'width':W}
    # nx.draw(G, **options)
    # plt.show()

    # ----------------------------------------------------------------------------
    # Notes
    # ----------------------------------------------------------------------------
    # G , colors= get_graph(cyTxt.spem.txt, cyTxt.spem.nodeColor)
    # from karateclub.node_embedding.neighbourhood import DeepWalk
    # model = DeepWalk(dimensions=2)
    # model.fit(G)
    # embedding = model.get_embedding()
    # print(embedding.shape)
    # print(G.number_of_nodes())
    #
    # x = embedding[:, 0]
    # y = embedding[:, 1]
    # c = [j for sub in colors for j in sub]
    #
    #
    # plt.scatter(x, y, c=colors)
    # plt.show()
    # for e in embedding:
    #     print(e)
