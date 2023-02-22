'''
The purpose of this script is to find all of the kcores for each of the graphs in 'graph_metadata' below. 
The kcores are stored in a list and pickled in the directory 'cores/'
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

'''
graph_metadata = [
	{"name": "karate", "edgelist_path": "../node2vec/graph/karate.edgelist"},
	{"name": "ppi", "edgelist_path": "../node2vec/graph/ppi.edgelist"},
	{"name": "wiki", "edgelist_path": "../node2vec/graph/wikipedia.edgelist"},
	{"name": "facebook", "edgelist_path": "../node2vec/graph/facebook_combined.edgelist"}
]
'''

'''
graph_metadata = [
	{"name": "enron", "edgelist_path": "../node2vec/graph/Email-Enron.edgelist"}
]
'''

graph_metadata = [
	{"name": "BTER-arbitrary", "edgelist_path": "../node2vec/graph/BTER_5000_arbitrary.edgelist"},
	{"name": "BTER-from-BA", "edgelist_path": "../node2vec/graph/BTER_from_BA_5000_5.edgelist"}
]
'''
graph_metadata = [
	{"name": "physics", "edgelist_path": "../node2vec/graph/ca-HepTh.edgelist"},
	{"name": "lastfm", "edgelist_path": "../node2vec/graph/lastfm_asia_edges.edgelist"},
	{"name": "BA-5", "edgelist_path": "../node2vec/graph/BA_5000_5.edgelist"},
	{"name": "BA-10", "edgelist_path": "../node2vec/graph/BA_5000_10.edgelist"},
	{"name": "ER-2", "edgelist_path": "../node2vec/graph/ER_5000_2.edgelist"},
	{"name": "ER-4", "edgelist_path": "../node2vec/graph/ER_5000_4.edgelist"}
]
'''
def test_core_pickle(name):
	'''
	Test that the kcores were successfully pickled (at least one core in the list)
	'''
	path = "cores/" + name + "_cores.pickle"
	with open(path, "rb") as pickleFile:
		assert(len(pickle.load(pickleFile)) > 0)

def find_cores(graph_nx):
	core_numbers = nx.algorithms.core.core_number(graph_nx)
	max_k = max(core_numbers.values())
	cores = []
	for k in range(1, max_k + 1):
		nodes = [node for node in core_numbers if core_numbers[node] >= k]
		cores.append(graph_nx.subgraph(nodes).copy())
	return cores

for graph_metadatum in graph_metadata:
	graph_nx = nx.read_edgelist(graph_metadatum["edgelist_path"])
	graph_nx.remove_edges_from(nx.selfloop_edges(graph_nx)) #remove self loops for kcore alg
	name = graph_metadatum["name"]
	print("Processing " + name)
	print("Nodes: " + str(len(graph_nx)))
	print("Edges: " + str(len(graph_nx.edges())))

	cores = find_cores(graph_nx)
	
	print("Found " + str(len(cores)) + " cores")
	path = "cores/" + name + "_cores.pickle"
	with open(path, "wb") as pickleFile:
		pickle.dump(cores, pickleFile)

	test_core_pickle(name)
