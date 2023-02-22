#import utils

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.core import core_number
import pickle
import base_line
import sklearn.preprocessing as skpp
from sklearn.manifold import SpectralEmbedding

from scipy.sparse.csgraph import laplacian

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter

import seaborn as sns

class DebiasModel:
    """
    debiasing the mining model
    """
    def __init__(self):
        self.error_reductions = []
        return

    def fit(self):
        return

    def get_error_reductions(self):
        return self.error_reductions

    def stable_line(self, name, graph, sim, stabilize, alpha=0.0,
             dimension=128, ratio=3200, negative=5,
             init_lr=0.025, batch_size=1000, seed=None, save_plot=False):
        """
        individually fair LINE
        :param graph: networkx nx.Graph()
        :param sim: similarity matrix
        :param alpha: regularization hyperparameter
        :param dimension: embedding dimension
        :param ratio: ratio to control edge sampling #sampled_edges = ratio * #nodes
        :param negative: number of negative samples
        :param init_lr: initial learning rate
        :param batch_size: batch size of edges in each training iteration
        :param seed: random seed
        :return: debiased node embeddings
        """
        if seed is not None:
            np.random.seed(seed)

        def _get_isolated_proximities(nx_graph, degenerate_nodes, dimension, seed):
            degenerate_core_nx = nx_graph.subgraph(degenerate_nodes)
            degenerate_core_mtx = nx.to_numpy_matrix(degenerate_core_nx,
                nodelist=degenerate_nodes)
            isolated_emb = base_line.LINE.embed(degenerate_core_nx, degenerate_core_mtx,
                dimension=dimension,
                ratio=250,
                batch_size = int(len(degenerate_nodes)/5))
            proximities = get_proximities(isolated_emb)
            return proximities

        def _base_error(emb):
            w = np.asarray(sim)
            return -np.sum(w * np.log(get_proximities(emb)))

        def _stability_error(emb):
            '''
                -alpha * sum_D [p_hat*log(p)]
            '''
            degenerate_emb = emb[[node2id[node] for node in degenerate_nodes]]
            degenerate_proximities = get_proximities(degenerate_emb)
            return alpha*np.sum((isolated_proximities - degenerate_proximities)**2)

        def _calculate_error(emb, idx, error):
            base_error = _base_error(emb)
            stability_error = _stability_error(emb)
            total_error = base_error + stability_error
            current_error = {"idx": idx,
                "base": base_error,
                "stability": stability_error,
                "total": total_error}
            #print(current_error)
            error.append(current_error)

        def _plot_error(error):
            idx = [iteration["idx"] for iteration in error]
            base = [iteration["base"] for iteration in error]
            stability = [iteration["stability"] for iteration in error]
            total = [iteration["total"] for iteration in error]

            plt.rc('axes', titlesize=20)     # fontsize of the axes title
            plt.rc('axes', titleweight="bold")     # fontweight of the axes title
            plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

            fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8,22))
            axs[0].set_title("Base (LINE)")
            axs[0].plot(idx, base)

            axs[1].set_title("Stability")
            axs[1].plot(idx, stability)

            axs[2].set_title("Total")
            axs[2].plot(idx, total)

            def thousands(x, pos):
                'The two args are the value and tick position'
                return '%1.1fK' % (x * 1e-3)
            for ax in axs:
                ax.yaxis.set_major_formatter(FuncFormatter(thousands))
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
            plt.savefig("plots/training_error_{}.jpg".format(name))

        def _update(vec_u, vec_v, vec_u_error, vec_v_error, label, is_degen_edge, u, v, lr, use_stability_penalty):
            f = 1 / (1 + np.exp(-np.sum(vec_u * vec_v, axis=1)))
            g = (lr * (label - f)).reshape((len(label), 1))
            vec_u_error -= g * -1*vec_v
            vec_v_error -= g * -1*vec_u

            if label[0] == 1 and use_stability_penalty:
                proximities = np.asarray([isolated_proximities[full_to_degen_id[u[i]]][full_to_degen_id[v[i]]]\
                    if is_degen_edge[i] else f[i] for i in range(len(is_degen_edge))])
                g = (lr * alpha * (f - proximities) * f * (1-f)).reshape((len(label), 1))
                vec_u_error -= g * vec_v
                vec_v_error -= g * vec_u

        def _train_line(use_stability_penalty=False, plot_error_bool=True):
            error = []
            for iter_num in range(nbatch):
                lr = init_lr * max((1 - iter_num * 1.0 / nbatch), 0.0001)
                u, v = [0] * batch_size, [0] * batch_size
                for i in range(batch_size):
                    edge_id = alias_draw(edges_table, edges_prob)
                    u[i], v[i] = edges[edge_id]
                    if not directed and np.random.rand() > 0.5:
                        v[i], u[i] = edges[edge_id]

                vec_u_error = np.zeros((batch_size, dimension))
                label = np.asarray([1 for _ in range(batch_size)])
                target = np.asarray(v)
                for j in range(negative + 1):
                    if j != 0: #determine whether u_j is a negative sample
                        label = np.asarray([0 for _ in range(batch_size)])
                        for k in range(batch_size):
                            target[k] = alias_draw(nodes_table, nodes_prob)
                    vec_v_error = np.zeros((batch_size, dimension))
                    is_degen_edge = [1 if (u[i] in degen_ids) and (target[i] in degen_ids) else 0 for i in range(batch_size)]
                    _update(
                        emb_vertex[u], emb_vertex[target], vec_u_error, vec_v_error, label, is_degen_edge, u, target, lr, use_stability_penalty
                    )
                    emb_vertex[target] += vec_v_error
                emb_vertex[u] += vec_u_error
                if plot_error_bool and iter_num % 100 == 0:
                    _calculate_error(emb_vertex, iter_num, error)
            if plot_error_bool:
                _plot_error(error)

        def _plot_stability_error(isolated_proximities, 
                original_emb, 
                emb_vertex, 
                degenerate_nodes):
            original_proximities = get_proximities(original_emb[[node2id[node] for node in degenerate_nodes]])
            final_proximities = get_proximities(emb_vertex[[node2id[node] for node in degenerate_nodes]])

            original_stability_err = np.ravel((original_proximities - isolated_proximities)**2)
            final_stability_err = np.ravel((final_proximities - isolated_proximities)**2)
            stability_error_pd = pd.DataFrame({
                "Stability Error": np.concatenate((original_stability_err, final_stability_err)),\
                "Algorithm": ["LINE"]*len(original_stability_err) + ["Stable LINE"]*len(final_stability_err)})
            fig, axs = plt.subplots()

            plt.rc('axes', titlesize=12)     # fontsize of the axes title
            plt.rc('axes', titleweight="bold")     # fontweight of the axes title
            plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=8)    # fontsize of the tick labels

            sns.kdeplot(data=stability_error_pd, x="Stability Error", hue="Algorithm", ax=axs, linewidth=3)
            axs.set_xscale("log")
            plt.savefig("plots/stability_error_distribution_{}.jpg".format(name))
            return np.mean(final_stability_err) / np.mean(original_stability_err)

        directed = nx.is_directed(graph)

        degenerate_nodes = get_degenerate_nodes(graph)

        nnodes = graph.number_of_nodes()
        node2id = dict([(node, vid) for vid, node in enumerate(graph.nodes())])
        degen_ids = [node2id[node] for node in degenerate_nodes]
        full_to_degen_id = dict([(degen_ids[i], i) for i in range(len(degen_ids))])

        edges = [[node2id[e[0]], node2id[e[1]]] for e in graph.edges()]
        edge_prob = np.asarray([graph[u][v].get("weight", 1.0) for u, v in graph.edges()])
        edge_prob /= np.sum(edge_prob)
        edges_table, edges_prob = alias_setup(edge_prob)

        degree_weight = np.asarray([0] * nnodes)
        for u, v in graph.edges():
            degree_weight[node2id[u]] += graph[u][v].get("weight", 1.0)
            if not directed:
                degree_weight[node2id[v]] += graph[u][v].get("weight", 1.0)
        node_prob = np.power(degree_weight, 0.75)
        node_prob /= np.sum(node_prob)
        nodes_table, nodes_prob = alias_setup(node_prob)

        nsamples = ratio * nnodes
        nbatch = int(nsamples / batch_size)
        emb_vertex = (np.random.random((nnodes, dimension)) - 0.5) / dimension

        # train
        if stabilize:
            isolated_proximities = _get_isolated_proximities(graph, degenerate_nodes, dimension, seed)
            nbatch = int(nbatch/2)
            _train_line(use_stability_penalty=False, plot_error_bool=False)
            original_emb = skpp.normalize(emb_vertex.copy(), "l2")
            _train_line(use_stability_penalty=True, plot_error_bool=save_plot)
            embeddings = skpp.normalize(emb_vertex, "l2")
            if save_plot:
                err_reduction = _plot_stability_error(isolated_proximities,
                    original_emb,
                    embeddings,
                    degenerate_nodes)
                self.error_reductions.append({
                    "Graph": name,
                    "Alg": "LINE",
                    "Alpha": alpha,
                    "dimension": dimension,
                    "ratio": ratio,
                    "negative": negative,
                    "init_lr": init_lr,
                    "batch_size": batch_size,
                    "Error Reduction": err_reduction
                    })
        else: 
            _train_line(use_stability_penalty=False, plot_error_bool=save_plot)
            embeddings = skpp.normalize(emb_vertex, "l2")

        # normalize
        return embeddings


    def stable_LE(self, name, graph, sim, alpha, beta,
             dimension=128, ratio=3200, negative=5,
             init_lr=0.025, batch_size=1000, seed=None, save_plot=False):
        """
        individually fair LINE
        :param graph: networkx nx.Graph()
        :param sim: similarity matrix
        :param alpha: regularization hyperparameter
        :param dimension: embedding dimension
        :param ratio: ratio to control edge sampling #sampled_edges = ratio * #nodes
        :param negative: number of negative samples
        :param init_lr: initial learning rate
        :param batch_size: batch size of edges in each training iteration
        :param seed: random seed
        :return: debiased node embeddings
        """
        if seed is not None:
            np.random.seed(seed)


        def _get_isolated_proximities(nx_graph, degenerate_nodes, dimension, seed):
            '''
                Embeds the degenerate core of nx_graph in isolation. 
                Returns a DxD matrix where the entry at i,j is the proximity btw 
                u_i and u_j where u are the embeddings of the isolated core. 
                The order of rows and columns is dicated by degenerate_nodes. 
            '''
            degenerate_core_nx = nx_graph.subgraph(degenerate_nodes)
            degenerate_core_adj = nx.to_numpy_array(degenerate_core_nx,
                nodelist=degenerate_nodes)
            isolated_emb = SpectralEmbedding(n_components=dimension,
                affinity="precomputed",
                n_jobs=-1,
                random_state=seed)\
            .fit_transform(degenerate_core_adj)
            proximities = get_proximities(isolated_emb)
            return proximities

        def _lap_lambda(core, d, seed):
            # return LaplacianEigenmaps(d=d).learn_embedding(core)[0]
            return SpectralEmbedding(n_components=d,
                                     affinity="precomputed",
                                     n_jobs=-1,
                                     random_state=seed)\
                    .fit_transform(nx.to_numpy_array(core))

        def _base_error(emb):
            '''
                base_error = tr(X'LX)
            '''
            L = lap_from_similarity(sim)
            return np.trace((emb.T @ L) @ emb)

        def _stability_error(emb):
            '''
                -alpha * sum_D [p_hat*log(p)]
            '''
            degenerate_emb = emb[[node2id[node] for node in degenerate_nodes]]
            degenerate_proximities = get_proximities(degenerate_emb)
            return alpha*np.sum((isolated_proximities - degenerate_proximities)**2)

        def _deviation_penalty(emb):
            return beta*np.linalg.norm(emb - original_emb, ord="fro")

        def _calculate_error(emb, idx, error):
            base_error = _base_error(emb)
            stability_error = _stability_error(emb)
            deviation_penalty = _deviation_penalty(emb)
            total_error = base_error + stability_error + deviation_penalty
            current_error = {"idx": idx,
                "base": base_error,
                "stability": stability_error,
                "deviation": deviation_penalty,
                "total": total_error}
            #print(current_error)
            error.append(current_error)

        def _plot_error(error):
            idx = [iteration["idx"] for iteration in error]
            base = [iteration["base"] + iteration["deviation"] for iteration in error]
            stability = [iteration["stability"] for iteration in error]
            total = [iteration["total"] for iteration in error]

            plt.rc('axes', titlesize=20)     # fontsize of the axes title
            plt.rc('axes', titleweight="bold")     # fontweight of the axes title
            plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

            fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8,22))
            axs[0].set_title("Base (LE)")
            axs[0].plot(idx, base)


            axs[1].set_title("Stability")
            axs[1].plot(idx, stability)
            #axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            axs[2].set_title("Total")
            axs[2].plot(idx, total)
            #axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            for ax in axs:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
            plt.savefig("plots/training_error_{}.jpg".format(name))

        def _update(vec_u, vec_v, vec_u_error, vec_v_error, label, u, v, lr, idx):
            ## L_base
            arr = np.asarray(sim[u, v].transpose())
            vec_u_error -= (2 * lr * (vec_u - vec_v) * arr)
            vec_v_error -= (2 * lr * (vec_v - vec_u) * arr)

            # L_stability
            f = 1 / (1 + np.exp(-np.sum(vec_u * vec_v, axis=1)))
            proximities = np.asarray([isolated_proximities[full_to_degen_id[u[i]]][full_to_degen_id[v[i]]]\
                if label[i] else f[i] for i in range(len(label))])
            #g = (lr * alpha * proximities * (1 - f)).reshape((len(label), 1))
            g = (lr * alpha * (f - proximities) * f * (1-f)).reshape((len(label), 1))
            vec_u_error -= g * vec_v
            vec_v_error -= g * vec_u

            # L_deviation
            vec_u_error -= lr * beta * (vec_u - original_emb[u])
            vec_v_error -= lr * beta * (vec_v - original_emb[v])

            # if idx % 100 == 0:
            #     print(g.shape)
            #     print(np.max(g))
            #     print(np.max(g * vec_v))
            #     print(np.max(g * vec_u))

        def _train():
            nbatch = int(nsamples / batch_size)
            error = [] # [{"Base":0.0, "Stability":0.0, "Total":0.0}]
            for iter_num in range(nbatch):
                lr = init_lr * max((1 - iter_num * 1.0 / nbatch), 0.0001)
                u, v = [0] * batch_size, [0] * batch_size
                for i in range(batch_size):
                    edge_id = alias_draw(edges_table, edges_prob)
                    u[i], v[i] = edges[edge_id]
                    if not directed and np.random.rand() > 0.5:
                        v[i], u[i] = edges[edge_id]

                vec_u_error = np.zeros((batch_size, dimension))
                vec_v_error = np.zeros((batch_size, dimension))
                label = np.asarray([1 if (u[i] in degen_ids) and (v[i] in degen_ids)
                    else 0
                    for i in range(batch_size)])
                temp = label.sum()
                target = np.asarray(v)
                _update(
                    emb_vertex[u], emb_vertex[target], vec_u_error, vec_v_error, label, u, target, lr, iter_num
                )
                emb_vertex[u] += vec_u_error
                emb_vertex[target] += vec_v_error
                if save_plot and iter_num % 100 == 0: 
                    _calculate_error(emb_vertex, iter_num, error)
            if save_plot:
                _plot_error(error)

        def _plot_stability_error(isolated_proximities, 
                original_emb, 
                emb_vertex, 
                degenerate_nodes):
            original_proximities = get_proximities(original_emb[[node2id[node] for node in degenerate_nodes]])
            final_proximities = get_proximities(emb_vertex[[node2id[node] for node in degenerate_nodes]])

            original_stability_err = np.ravel((original_proximities - isolated_proximities)**2)
            final_stability_err = np.ravel((final_proximities - isolated_proximities)**2)
            stability_error_pd = pd.DataFrame({
                "Stability Error": np.concatenate((original_stability_err, final_stability_err)),\
                "Algorithm": ["Laplacian Eigenmaps"]*len(original_stability_err) + ["Stable LE"]*len(final_stability_err)})
            fig, axs = plt.subplots()

            plt.rc('axes', titlesize=12)     # fontsize of the axes title
            plt.rc('axes', titleweight="bold")     # fontweight of the axes title
            plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=8)    # fontsize of the tick labels

            sns.kdeplot(data=stability_error_pd, x="Stability Error", hue="Algorithm", ax=axs, linewidth=3)
            #axs.set_title("Distribution of Stability Errors by Algorithm")
            axs.set_xscale("log")
            plt.savefig("plots/stability_error_distribution_{}.jpg".format(name))
            return np.mean(final_stability_err) / np.mean(original_stability_err)

        directed = nx.is_directed(graph)
        degenerate_nodes = get_degenerate_nodes(graph)
        # Matrix of proximities for isolated degenerate core 
        isolated_proximities = _get_isolated_proximities(graph, degenerate_nodes, dimension, seed)
        #print(isolated_proximities)
        # Augment graph with degenerate core edges
        original_graph = graph.copy() 
        augment_graph(graph, degenerate_nodes)

        nnodes = graph.number_of_nodes()
        node2id = dict([(node, vid) for vid, node in enumerate(graph.nodes())])
        degen_ids = [node2id[node] for node in degenerate_nodes]
        full_to_degen_id = dict([(degen_ids[i], i) for i in range(len(degen_ids))])

        edges = [[node2id[e[0]], node2id[e[1]]] for e in graph.edges()]
        edge_prob = np.asarray([graph[u][v].get("weight", 1.0) for u, v in graph.edges()])
        edge_prob /= np.sum(edge_prob)
        edges_table, edges_prob = alias_setup(edge_prob)

        nsamples = ratio * nnodes
        emb_vertex = _lap_lambda(original_graph, dimension, seed)
        original_emb = emb_vertex.copy()

        # train
        _train()
        if save_plot: 
            err_reduction = _plot_stability_error(isolated_proximities,
                original_emb,
                emb_vertex,
                degenerate_nodes)
            self.error_reductions.append({
                "Graph": name,
                "Alg": "LE",
                "Alpha": alpha,
                "Beta": beta,
                "dimension": dimension,
                "ratio": ratio,
                "negative": negative,
                "init_lr": init_lr,
                "batch_size": batch_size,
                "Error Reduction": err_reduction
                })
        return emb_vertex


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

    '''
        Error = Tr(X'LX) - alpha * (sum_D p_hat * log(p))

        Base: X (emb) L (sim)
        Stability: p_hat (isolated_proximities) p (emb, degenerate_nodes, )
    '''

def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

def get_degenerate_nodes(nx_graph):
    node_to_core = core_number(nx_graph)
    degeneracy = max(node_to_core.values())
    degenerate_nodes = []
    for node in nx_graph:
        if node_to_core[node] == degeneracy:
            degenerate_nodes.append(node)
    return degenerate_nodes

def lap_from_similarity(sim):
    '''
    Return the laplacian given the similarity/adjacency matrix 
    '''
    return np.diag(np.asarray(sim.sum(axis=0))[0]) - sim

def get_proximities(emb):
    proximities = 1 / (1 + np.exp(-emb @ emb.T))
    np.fill_diagonal(proximities, 1)
    return proximities

def export_emb(emb, description, filename):
    with open("embeddings/" + filename, "wb") as pickleFile:
        pickle.dump((description, emb), pickleFile)

def augment_graph(nx_graph, degenerate_nodes):
    '''
        Adds an edge between every pair of nodes in the degenerate core of nx_graph
        This ensures that negative degenerate-core edges are sampled during training
        Assumes that if an edge exists, calling add_edge is a no-op
    '''
    for i in range(len(degenerate_nodes)):
        for j in range(i+1, len(degenerate_nodes)):
            nx_graph.add_edge(degenerate_nodes[i], degenerate_nodes[j])

if __name__ == "__main__":
    graph_metadata = [
        # {
        # "name": "Enron",
        # "edgelist": "../../node2vec/graph/Email-Enron.edgelist"
        # },
        {
        "name": "Facebook",
        "edgelist": "../../node2vec/graph/facebook_combined.edgelist"
        },
        {
        "name": "Autonomous Systems",
        "edgelist": "../../node2vec/graph/AS.edgelist"
        },
        {
        "name": "Protein-Protein",
        "edgelist": "../../node2vec/graph/ppi.edgelist"
        },
        {
        "name": "ca-HepTh",
        "edgelist": "../../node2vec/graph/ca-HepTh.edgelist"
        },
        {
        "name": "LastFM",
        "edgelist": "../../node2vec/graph/lastfm_asia_edges.edgelist"
        },
        {
        "name": "Wikipedia",
        "edgelist": "../../node2vec/graph/wikipedia.edgelist"
        },
    ]
    FairModel = DebiasModel()
    for graph_metadatum in graph_metadata:
        graph = nx.read_edgelist(graph_metadatum["edgelist"])
        graph.remove_edges_from(nx.selfloop_edges(graph))
        sim = nx.to_numpy_matrix(graph)
        stabilize = True
        beta = 1.0
        dimension = 32
        ratio = 1000
        #batch_size = 1000
        batch_size = int(len(graph)/5)

        #alpha = 1000000
        alphas = [1e3, 1e4, 1e5]
        if graph_metadatum["name"] == "Facebook":
            alphas.append(1e6)

        for alpha in alphas:
            graph_title = "{}_stable_LE_{}".format(graph_metadatum["name"], alpha)
            embs = FairModel.stable_LE(
                graph_title, graph, sim, alpha, beta,
                dimension=dimension, ratio=ratio, negative=5,
                init_lr=0.025, batch_size=batch_size, seed=0, save_plot=True
            )

        alphas = [1, 10, 100]
        for alpha in alphas:
            graph_title = "{}_stable_LINE_{}".format(graph_metadatum["name"], alpha)
            embs = FairModel.stable_line(
                graph_title, graph, sim, stabilize, alpha,
                dimension=dimension, ratio=ratio, negative=5,
                init_lr=0.025, batch_size=batch_size, seed=0, save_plot=True
            )


    with open("error_reductions_d32.pickle", "wb") as pickleFile:
        pickle.dump(FairModel.get_error_reductions(), pickleFile)

        # description = "Individually Fair LINE Embeddings for Facebook graph\
        #     alpha fairness parameter = 0.5\
        #     dimension = 64\
        #     all other paramters are out of the box."
        # export_emb(embs, description, "Stable_FB.embeddings")


