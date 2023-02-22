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
		"alpha" : {
			"Stable LE": 1e6,
			"Stable LINE": 100
		}
		},
		{
		"name": "Autonomous Systems",
		"edgelist": "../../node2vec/graph/AS.edgelist",
		"alpha" : {
			"Stable LE": 1e5,
			"Stable LINE": 100
		}
		},
		{
		"name": "Protein-Protein",
		"edgelist": "../../node2vec/graph/ppi.edgelist",
		"alpha" : {
			"Stable LE": 1e5,
			"Stable LINE": 100
		}
		},
		{
		"name": "ca-HepTh",
		"edgelist": "../../node2vec/graph/ca-HepTh.edgelist",
		"alpha" : {
			"Stable LE": 1e5,
			"Stable LINE": 100
		}
		},
		{
		"name": "LastFM",
		"edgelist": "../../node2vec/graph/lastfm_asia_edges.edgelist",
		"alpha" : {
			"Stable LE": 1e5,
			"Stable LINE": 100
		}
		},
		{
		"name": "Wikipedia",
		"edgelist": "../../node2vec/graph/wikipedia.edgelist",
		"alpha" : {
			"Stable LE": 1e5,
			"Stable LINE": 100
		}
		},
	]
	algorithms = [
		{"name": "LE", "lambda": embed_cores.lap_lambda},
		{"name": "Stable LE", "lambda": embed_cores.stable_lap_lambda},
		{"name": "LINE", "lambda": embed_cores.line_lambda},
		{"name": "Stable LINE", "lambda": embed_cores.stable_line_lambda}
	]

	d = 32
	iterations = 15
	results = []

	start_evaluation = time.time()
	for graph_metadatum in graph_metadata:
		G = nx.read_edgelist(graph_metadatum["edgelist"])
		G.remove_edges_from(nx.selfloop_edges(G)) #remove self loops for kcore alg
		A = nx.to_scipy_sparse_matrix(G, dtype=float)

		for iteration in range(iterations):
			data_splits = utils.train_val_test_split(A)

			training_graph = nx.from_scipy_sparse_matrix(data_splits["adjacency_train"],
				create_using=nx.Graph(),
				edge_attribute="weight")
			for alg in algorithms:
				#print(graph_metadatum["name"], alg["name"])
				start_embedding = time.time()
				if alg["name"] in graph_metadatum["alpha"]:
					emb = alg["lambda"]("NA", training_graph, d, graph_metadatum["alpha"][alg["name"]])
				else:
					emb = alg["lambda"]("NA", training_graph, d)
				end_embedding = time.time()

				roc, f1 = utils.link_prediction(data_splits, emb)
				results.append({
					"Graph": graph_metadatum["name"],
					"Algorithm": alg["name"],
					"Runtime": end_embedding - start_embedding,
					"ROC": roc,
					"F1": f1,
					"iteration": iteration
					})
				print("Graph: {} Algorithm: {} Iteration: {} ROC: {} F1: {}".format(
					graph_metadatum["name"],
					alg["name"],
					iteration,
					roc, f1))
	end_evaluation = time.time()
	results_pd = pd.DataFrame(results)
	results_pd.to_csv("link-prediction-result-large.csv")
	print(results_pd)
	print("Elapsed {} seconds".format(end_evaluation - start_evaluation))

