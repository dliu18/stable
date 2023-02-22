import pickle
import json
import networkx as nx
import utils 
import pickle 
import numpy as np
import subprocess
import sys

try:
    from gem.evaluation import visualize_embedding as viz
    from gem.embedding.hope import HOPE
    #from gem.embedding.sdne	    import SDNE
    from sklearn.manifold import SpectralEmbedding
    from sklearn.decomposition import TruncatedSVD, PCA
    from networkx.algorithms.core import core_number
except:
    pass

try:
    from ge import SDNE
except:
    pass

import debias_model
import base_line

#MACROS
MAX_TRIES = 3
random_state = 42

###########################################################################
#LAMBDAS:
#Input: nx.Graph object corresponding to core and embedding dimension d
#Output: np.ndarray embedding matrix where rows are in the nodelist order specified by cores.nodes()

def hope_lambda(graphName, core, d):
    return HOPE(d=d, beta=0.01).learn_embedding(core)[0]

def lap_lambda(graphName, core, d):
    # return LaplacianEigenmaps(d=d).learn_embedding(core)[0]
    return SpectralEmbedding(n_components=d,
                             affinity="precomputed",
                             n_jobs=-1,
                             random_state=random_state)\
            .fit_transform(nx.linalg.graphmatrix.adjacency_matrix(core))

def stable_lap_lambda(graphName, core, d, alpha=1e5):
    FairModel = debias_model.DebiasModel()
    beta = 1.0
    ratio = 1000
    batch_size = int(len(core)/5)
    return FairModel.stable_LE(
        graphName, core, nx.to_numpy_matrix(core), alpha, beta,
        d, ratio, batch_size=batch_size, seed=random_state)

def line_lambda(graphName, core, d):
    return base_line.LINE.embed(core,
        nx.to_numpy_matrix(core),
        dimension=d,
        ratio=1000,
        batch_size=int(len(core)/5))

def stable_line_lambda(graphName, core, d, alpha=100):
    batch_size=int(len(core)/5)
    ratio=1000
    FairModel = debias_model.DebiasModel()
    return FairModel.stable_line(graphName,
        core,
        nx.to_numpy_matrix(core),
        stabilize=True,
        alpha=alpha,
        dimension=d,
        ratio=ratio,
        batch_size=batch_size,
        seed=random_state,
        save_plot=False)

def sdne_lambda(graphName, core, d):
#     return SDNE(d=d, 
#                 beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, 
#                 K=3,n_units=[1000, 500,], 
#                 rho=0.3, 
#                 n_iter=30, 
#                 xeta=0.01,
#                 n_batch=1024,
#                 modelfile=['enc_model.json', 'dec_model.json'],
#                 weightfile=['enc_weights.hdf5', 'dec_weights.hdf5'])\
#             .learn_embedding(core)[0]
    model = SDNE(core, hidden_size=[256, d], alpha=0.1, beta=10)
    model.train(epochs=100)
    emb_dict = model.get_embeddings()
    return np.array([emb_dict[node_name] for node_name in core.nodes()])

def pca_lambda(graphName, core, d):
    return TruncatedSVD(n_components=d, random_state=random_state)\
            .fit_transform(nx.linalg.graphmatrix.adjacency_matrix(core))

def read_embedding():
    with open("temp.emb", "r") as embeddingFile:
        embeddings = {}
        for line in embeddingFile:
            line_entries = line.split(" ")
            if len(line_entries) == 2:
                continue #skip header
            embeddings[line_entries[0]] =\
                np.array([float(coord) for coord in line_entries[1:]])
    return embeddings

def n2v_lambda(graphName, core, d): 
    nx.write_edgelist(core, "temp.edgelist")
    subprocess.run(["python",\
            "../node2vec/src/main.py",\
            "--input", "temp.edgelist",\
            "--output", "temp.emb",\
            "--dimension", str(d),\
            "--workers", "24"])
    embeddings = read_embedding()
    return np.array([embeddings[node] for node in core.nodes()])

def hgcn_lambda(graphName, core, d):
    subprocess.run(["rm", "-rf", "hgcn/data/{}/{}.edgelist".format(graphName, graphName)])
    subprocess.run(["rm", "-rf", "hgcn/logs/{}/embeddings.npy".format(graphName)])
    
    nx.write_edgelist(core, "hgcn/data/{}/{}.edgelist".format(graphName, graphName))
    result = subprocess.run(["python",
                   "hgcn/train.py",
                   "--task", "lp",
                    "--dataset", graphName,
                    "--save", "1",
                    "--save-dir", "hgcn/logs/" + graphName,
                    "--model", "Shallow",
                    "--manifold", "PoincareBall",
                    "--lr", "0.01",
                    "--weight-decay", "0.0005",
                    "--dim", str(d),
                    "--num-layers", "0",
                    "--use-feats", "0",
                    "--dropout", "0.2",
                    "--act", "None",
                    "--bias", "0",
                    "--optimizer", "RiemannianAdam",
                    "--cuda", "0",
                    "--log-freq", "1000"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    return np.load("hgcn/logs/{}/embeddings.npy".format(graphName))

def get_embedding_algorithms(algNames):
    algLambdas = []
    if "sdne" in algNames:
        algLambdas.append({"name": "sdne", "method": sdne_lambda})
    if "pca" in algNames:
        algLambdas.append({"name": "pca", "method": pca_lambda})
    if "HOPE" in algNames:
        algLambdas.append({"name": "HOPE", "method": hope_lambda})
    if "lap" in algNames:
        algLambdas.append({"name": "lap", "method": lap_lambda})
    if "n2v" in algNames:
        algLambdas.append({"name": "n2v", "method": n2v_lambda})
    if "hgcn" in algNames:
        algLambdas.append({"name": "hgcn", "method": hgcn_lambda})
    if "stable-lap" in algNames:
        algLambdas.append({"name": "stable-lap", "method": stable_lap_lambda})
    return algLambdas

###########################################################################
#HELPERS

def load_kcores(graphName):
    with open("cores/" + graphName + "_cores.pickle", "rb") as pickleFile:
        cores = pickle.load(pickleFile)
    assert len(cores) >= 1
    return cores

def embed_cores(graphName, cores, d, algs_to_attempt, subset=True):
    embeddings = {}
    cores_subset = utils.get_cores_subset(cores)
    if not subset:
        cores_subset = [(i+1, cores[i]) for i in range(len(cores))]
        
    successful_alg_objs = []
    for alg in algs_to_attempt: 
        print(alg["name"] + " " + str(d))
        method = alg["method"]
#        try: 
        embeddings[alg["name"]] = [(k, method(graphName, core, d)) for k, core in cores_subset]
        successful_alg_objs.append(alg["name"])
#         except:
#             print("Embedding Alg {} failed with d = {}".format(alg, str(d)))
#             continue
    return (successful_alg_objs, embeddings) 
 
def export_embeddings(parent_dir, graphName, all_embeddings):
    with open(parent_dir + graphName + "_embeddings.pickle", "wb") as pickleFile: 
        pickle.dump(all_embeddings, pickleFile)
        
def import_embeddings(parent_dir, graphName, d_s):
    try:
        with open(parent_dir + graphName + "_embeddings.pickle", "rb") as pickleFile:
            return pickle.load(pickleFile)	
    except:
        #file does not exist, create emtpy embedding dictionary
        all_embeddings = {}
        for d in d_s:
            all_embeddings[str(d)] = [[], {}]
        return all_embeddings

def completed_all_embeddings(completed_algs, all_embedding_algorithms):
    for alg in all_embedding_algorithms:
        if alg["name"] not in completed_algs:
            return False
    return True

def main(parent_dir, graphNames, algNames, d_s, override, subset):
    for graphName in graphNames:
        print(graphName)
        all_embeddings = import_embeddings(parent_dir, graphName, d_s) 
        cores = load_kcores(graphName)
        for d in d_s:
            attempt = 1
            embedding_algorithms = get_embedding_algorithms(algNames)
            if str(d) not in all_embeddings:
                all_embeddings[str(d)] = [[], {}]
      
            if override: 
                for alg in embedding_algorithms:
                    if alg["name"] in all_embeddings[str(d)][0]:
                        all_embeddings[str(d)][0].remove(alg["name"])
                        all_embeddings[str(d)][1].pop(alg["name"], None)
                    
            print(all_embeddings[str(d)][0])       
            while attempt <= MAX_TRIES and not completed_all_embeddings(all_embeddings[str(d)][0], embedding_algorithms):
                print("Graph {} d={} Attempt {}".format(graphName, str(d), str(attempt)))
                algs_to_attempt = []
                for alg in embedding_algorithms: 
                    if alg["name"] not in all_embeddings[str(d)][1]:
                        algs_to_attempt.append(alg)
                        print("Trying {}".format(alg["name"]))
                embedding_output = embed_cores(graphName, cores, d, algs_to_attempt, subset)
                all_embeddings[str(d)][0].extend(embedding_output[0])
                all_embeddings[str(d)][1].update(embedding_output[1])
                attempt+=1   
            export_embeddings(parent_dir, graphName, all_embeddings)

###########################################################################
#MAIN

if __name__ == "__main__":
    #Read Config File
    try:
        assert len(sys.argv) > 1
        print(sys.argv)
        with open(sys.argv[1], "r") as configFile:
            config = json.load(configFile)
    except:
        print("Usage: python embed_cores.py config_file.json")
        exit()
    main(**config)
