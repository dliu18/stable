import numpy as np
from scipy.sparse import diags, isspmatrix_coo, triu

import networkx as nx

from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
import embed_cores
# Convert sparse matrix to tuple
def sparse_to_tuple(mat):
    if not isspmatrix_coo(mat):
        mat = mat.tocoo()
    coords = np.vstack((mat.row, mat.col)).transpose()
    values = mat.data
    shape = mat.shape
    return coords, values, shape

def train_val_test_split(A, test_frac=.1, val_frac=.05, prevent_disconnect=False, is_directed=False, false_ratio=1):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    result = dict()

    # graph should not have diagonal values
    if is_directed:
        G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph(), edge_attribute='weight')
    else:
        G = nx.from_scipy_sparse_matrix(A, create_using=nx.Graph(), edge_attribute='weight')
    num_cc = nx.number_connected_components(G)

    A_triu = triu(A) # upper triangular portion of adj matrix
    A_tuple = sparse_to_tuple(A_triu) # (coords, values, shape), edges only 1 way
    edges = A_tuple[0] # all edges, listed only once (not 2 ways)
    num_test = int(np.floor(edges.shape[0] * test_frac)) # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac)) # controls how alrge the validation set should be

    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples) # initialize train_edges to have all edges
    test_edges, val_edges = set(), set()

    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        node1, node2 = edge[0], edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on
        G.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if nx.number_connected_components(G) > num_cc:
                G.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    #print("Created positive edges")
    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: ({}, {})".format(num_test, num_val))
        print("Num. (test, val) edges returned: ({}, {})".format(len(test_edges), len(val_edges)))

    if prevent_disconnect == True:
        assert nx.number_connected_components(G) == num_cc

    test_edges_false = set()
    while len(test_edges_false) < num_test * false_ratio:
        idx_i, idx_j = np.random.randint(0, A.shape[0]), np.random.randint(0, A.shape[0])
        if idx_i == idx_j:
            continue
        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))
        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue
        test_edges_false.add(false_edge)
    #print("Created test false edges")

    val_edges_false = set()
    while len(val_edges_false) < num_val * false_ratio:
        idx_i = np.random.randint(0, A.shape[0])
        idx_j = np.random.randint(0, A.shape[0])
        if idx_i == idx_j:
            continue
        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false:
            continue
        val_edges_false.add(false_edge)
    #print("Created val false edges")

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges) * false_ratio:
        idx_i = np.random.randint(0, A.shape[0])
        idx_j = np.random.randint(0, A.shape[0])
        if idx_i == idx_j:
            continue
        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, 
            # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false or \
            false_edge in train_edges_false:
            continue
        train_edges_false.add(false_edge)
    #print("Created train false edges")

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    # Convert edge-lists to numpy arrays
    result['adjacency_train'] = nx.adjacency_matrix(G)
    result['train_edge_pos'] = np.array([list(edge_tuple) for edge_tuple in train_edges])
    result['train_edge_neg'] = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    result['val_edge_pos'] = np.array([list(edge_tuple) for edge_tuple in val_edges])
    result['val_edge_neg'] = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    result['test_edge_pos'] = np.array([list(edge_tuple) for edge_tuple in test_edges])
    result['test_edge_neg'] = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    # NOTE: these edge lists only contain single direction of edge!
    return result

def get_score(embs, src, tgt):
    """
    calculate score for link prediction
    :param embs: embedding matrix
    :param src: source node
    :param tgt: target node
    """
    vec_src = embs[int(src)]
    vec_tgt = embs[int(tgt)]
    return np.dot(vec_src, vec_tgt) / (np.linalg.norm(vec_src) * np.linalg.norm(vec_tgt))


def link_prediction(data, embs):
    """
    link prediction
    :param data: input data
    :param embs: embedding matrix
    """
    true_edges = data['test_edge_pos']
    false_edges = data['test_edge_neg']

    true_list = list()
    prediction_list = list()
    for src, tgt in true_edges:
        true_list.append(1)
        prediction_list.append(get_score(embs, src, tgt))

    for src, tgt in false_edges:
        true_list.append(0)
        prediction_list.append(get_score(embs, src, tgt))

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-len(true_edges)]

    ypred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            ypred[i] = 1

    ytrue = np.array(true_list)
    yscores = np.array(prediction_list)
    precision, recall, _ = precision_recall_curve(ytrue, yscores)
    return roc_auc_score(ytrue, yscores), f1_score(ytrue, ypred)

## wrapper function so that the embedding lambda ("lam") can be called in parallel
def embed_parallel(graphName, d, lam, G):
    start_embedding = time.time()
    emb = lam(graphName, G, d)
    return time.time() - start_embedding, emb