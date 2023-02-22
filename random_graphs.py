import networkx as nx

#BA graph with n = 10000 m = 5
ba_graph = nx.generators.random_graphs.barabasi_albert_graph(n=5000, m=5)
assert(len(ba_graph) == 5000)
#assert(len(ba_graph.edges()) == 25000)
nx.write_edgelist(ba_graph, "../node2vec/graph/BA_5000_5.edgelist")

#BA graph with n = 10000 m = 10
ba_graph = nx.generators.random_graphs.barabasi_albert_graph(n=5000, m=10)
assert(len(ba_graph) == 5000)
#assert(len(ba_graph.edges()) == 50000)
nx.write_edgelist(ba_graph, "../node2vec/graph/BA_5000_10.edgelist")

#ER graph p = 0.002
er_graph = nx.generators.random_graphs.erdos_renyi_graph(n=5000, p=0.002, seed=1)
assert(len(er_graph) == 5000)
nx.write_edgelist(er_graph, "../node2vec/graph/ER_5000_2.edgelist")


#ER graph p = 0.004
er_graph = nx.generators.random_graphs.erdos_renyi_graph(n=5000, p=0.004, seed=1)
assert(len(er_graph) == 5000)
nx.write_edgelist(er_graph, "../node2vec/graph/ER_5000_4.edgelist")
