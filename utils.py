import numpy as np

def get_cores_subset(cores):
        subset = []
        subsetSize = 5
        if len(cores) <= subsetSize:
            subset = [(i+1, cores[i]) for i in range(len(cores))]
        else:
            subset.append((1, cores[0]))
            for i in range(1, subsetSize):
                    idx = int(i*len(cores)/subsetSize)
                    subset.append((idx+1, cores[idx]))
            subset.append((len(cores), cores[-1]))
        return subset

def delta(features):
    return np.array(features[1:]) - np.array(features[:-1])

def normalize(values):
    return (np.array(values) - min(values))/max(values)

def core_completeness(core):
    num_edges = len(core.edges())
    num_nodes = len(core.nodes())
    max_num_edges = 0.5 * num_nodes*(num_nodes - 1)
    return num_edges / max_num_edges