import networkx as nx 
import numpy as np 
import pandas as pd
import scipy.sparse 
import pickle 

import utils
import embed_cores

import time

if __name__ == "__main__":

	# Generate train test splits
	graph_metadata = [
		# {
		# "name": "Enron",
		# "edgelist": "../../node2vec/graph/Email-Enron.edgelist"
		# },
		{
		"name": "Facebook",
		"edgelist": "../../node2vec/graph/facebook_combined.edgelist",
		"cores": "../cores/facebook_cores.pickle",
		"embeddings": "../embeddings_all_cores/facebook_embeddings.pickle"
		},
		{
		"name": "Autonomous Systems",
		"edgelist": "../../node2vec/graph/AS.edgelist",
		"cores": "../cores/AS_cores.pickle",
		"embeddings": "../embeddings_all_cores/AS_embeddings.pickle"
		},
		{
		"name": "Protein-Protein",
		"edgelist": "../../node2vec/graph/ppi.edgelist",
		"cores": "../cores/ppi_cores.pickle",
		"embeddings": "../embeddings_all_cores/ppi_embeddings.pickle"
		},
		{
		"name": "ca-HepTh",
		"edgelist": "../../node2vec/graph/ca-HepTh.edgelist",
		"cores": "../cores/physics_cores.pickle",
		"embeddings": "../embeddings_all_cores/physics_embeddings.pickle"
		},
		{
		"name": "LastFM",
		"edgelist": "../../node2vec/graph/lastfm_asia_edges.edgelist",
		"cores": "../cores/lastfm_cores.pickle",
		"embeddings": "../embeddings_all_cores/lastfm_embeddings.pickle"
		},
		{
		"name": "Wikipedia",
		"edgelist": "../../node2vec/graph/wikipedia.edgelist",
		"cores": "../cores/wiki_cores.pickle",
		"embeddings": "../embeddings_all_cores/wiki_embeddings.pickle"
		},
	]
	algorithms = ["lap"]

	d_str = "10"
	iterations = 1
	results = []

	start_evaluation = time.time()
	for graph_metadatum in graph_metadata:
		with open(graph_metadatum["cores"], "rb") as pickleFile:
			cores = pickle.load(pickleFile)
		with open(graph_metadatum["embeddings"], "rb") as pickleFile:
			emb = pickle.load(pickleFile)

		degenerate_core = cores[-1]
		degenerate_core.remove_edges_from(nx.selfloop_edges(degenerate_core)) #remove self loops for kcore alg
		# A_degenerate = nx.to_scipy_sparse_matrix(degenerate_core, dtype=float)
		# data_splits = utils.train_val_test_split(A_degenerate, test_frac=0.9, false_ratio=0.1)


		for k in range(1, len(cores) + 1):
			core = cores[k - 1]
			core.remove_edges_from(nx.selfloop_edges(core)) #remove self loops for kcore alg
			
			# degenerate_idxs = []
			# for node in degenerate_core:
			# 	for i in range(len(core)):
			# 		if list(core.nodes())[i] == node:
			# 			degenerate_idxs.append(i)
			# 			break
			A_core = nx.to_scipy_sparse_matrix(core, dtype=float)
			data_splits = utils.train_val_test_split(A_core, test_frac=0.9, false_ratio=0.1)
			
			for alg in algorithms:
					#print(graph_metadatum["name"], alg["name"])
					_ , core_emb = emb[d_str][1][alg][k - 1]
					#degenerate_emb = core_emb[degenerate_idxs]
					#roc, f1 = utils.link_prediction(data_splits, degenerate_emb)
					roc, f1 = utils.link_prediction(data_splits, core_emb)

					results.append({
						"Graph": graph_metadatum["name"],
						"Algorithm": alg,
						"k": k,
						"ROC": roc,
						"F1": f1,
						"Size": len(core) / len(cores[0])
						})
					print("Graph: {} Algorithm: {} K: {} ROC: {} F1: {}".format(
						graph_metadatum["name"],
						alg,
						k,
						roc, f1))
					results_pd = pd.DataFrame(results)
					results_pd.to_csv("link-prediction-toy.csv")
	end_evaluation = time.time()
	print(results_pd)
	print("Elapsed {} seconds".format(end_evaluation - start_evaluation))

