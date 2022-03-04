import node2vec
import numpy as np
import networkx as nx
from gensim.models import Word2Vec



def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight',float),),
        create_using=nx.DiGraph())

    return G

def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=dimensions, window = 10, min_count=0, sg=1,
        workers = 8, epochs = iter)
    model.wv.save_word2vec_format(output_file)
	
def generate_adj_mx(output_file):
    import pickle
    file = open('../data/sample.pickle', 'rb')
    _, d, _ = pickle.load(file)
    file.close()

    file = open(output_file, 'w+')
    cnt = 0
    for i in range(N):
        for j in range(N):
            file.write(f'{i} {j} {np.round(d[cnt],3)}\n')
            cnt += 1
    file.close()



if __name__ == "__main__":
    is_directed = False
    p = 2
    q = 1
    N = 30
    num_walks = 100
    walk_length = 80
    dimensions = 64
    window_size = 10
    iter = 1000
    Adj_file = 'data/Adj.txt'
    SE_file = 'data/SE.txt'

    # Generate adjancency matrix
    generate_adj_mx(Adj_file)

    nx_G = read_graph(Adj_file)
    G = node2vec.Graph(nx_G, is_directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    learn_embeddings(walks, dimensions, SE_file)
