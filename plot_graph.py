import ampligraph
import numpy as np
import pandas as pd
import tensorflow as tf
from ampligraph.datasets import load_fb15k_237
from ampligraph.evaluation import train_test_split_no_unseen, evaluate_performance, mr_score, mrr_score, hits_at_n_score
from ampligraph.discovery import query_topn, discover_facts, find_clusters
from ampligraph.latent_features import TransE, ComplEx, HolE, DistMult, ConvE, ConvKB
from ampligraph.utils import save_model, restore_model
from graphviz import Digraph

def combine_same_node(node_name, node_seq):
    for node1, indexI in zip(node_name, range(0, len(node_name))):
        for indexJ in range(indexI + 1, len(node_name) - 1):
            if node1 == node_name[indexJ]:
                node_seq[indexJ] = node_seq[indexI]

    return node_seq


URL = 'https://ampgraphenc.s3-eu-west-1.amazonaws.com/datasets/freebase-237-merged-and-remapped.csv'
dataset = pd.read_csv(URL, header=None)
dataset.columns = ['subject', 'predicate', 'object']

# get the validation set of size 500
test_train, X_valid = train_test_split_no_unseen(dataset.values, 500, seed=0)

# get the test set of size 1000 from the remaining triples
X_train, X_test = train_test_split_no_unseen(test_train, 1000, seed=0)

node_1 = X_train[0:20, 0].tolist()
node_2 = X_train[0:20, 2].tolist()
graph_edge = X_train[0:20, 1].tolist()


node_seq_1 = ['node_1' + str(index) for index in range(0,len(node_1))]
node_seq_2 = ['node_2' + str(index) for index in range(0,len(node_2))]
# print(node_seq_1, node_seq_2)

node_seq_1 = combine_same_node(node_1, node_seq_1)
node_seq_2 = combine_same_node(node_2, node_seq_2)
# print(node_seq_1, node_seq_2)



g = Digraph('G', filename='hello.gv')

for node1, node2, indexI in zip(node_1, node_2, range(0,len(node_1))):
    g.node(node_seq_1[indexI], label = node1)
    g.node(node_seq_2[indexI], label = node2)

    g.edge(node_seq_1[indexI], node_seq_2[indexI], graph_edge[indexI])


g.view()