import networkx as nx
from karateclub import FeatherNode
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.stats import entropy

#Using the karateclub(https://karateclub.readthedocs.io/en/latest/_modules/karateclub/node_embedding/attributed/feathernode.html) implementation
#of  https://arxiv.org/abs/2005.07959
def calcul_graph_embed_feather(graphs, labels):
    dimensions = 64
    epochs = 10
    embeddings = []

    filtered = [(G, lbl) for G, lbl in zip(graphs, labels) if G.number_of_nodes() > 0]
    if len(filtered) < len(graphs):
        dropped = len(graphs) - len(filtered)
        print(f"Warning: dropping {dropped} empty graph(s) before embedding")
    graphs, labels = zip(*filtered)

    for graph in graphs:
        model = FeatherNode(reduction_dimensions=dimensions, svd_iterations=epochs)

        feature_dict = nx.get_node_attributes(graph, 'mean_activation')
        if not feature_dict:
            raise ValueError("Graph has no node features")
        X = np.array(list(feature_dict.values())).reshape(-1, 1)

        model.fit(graph, X=X)
        #Aggregating the node embeddings using statistical pooling to obtain the graph embeddings
        node_embeddings = model.get_embedding()
        mean_embed = node_embeddings.mean(axis=0)
        max_embed = node_embeddings.max(axis=0)
        min_embed = node_embeddings.min(axis=0)
        median_embed = np.median(node_embeddings, axis=0)
        range_embed = max_embed - node_embeddings.min(axis=0)
        skewness_embed = skew(node_embeddings, axis=0)
        kurtosis_embed = kurtosis(node_embeddings, axis=0)
        q75 = np.percentile(node_embeddings, 75, axis=0)
        q25 = np.percentile(node_embeddings, 25, axis=0)
        iqr_embed = q75 - q25
        prob_embed = node_embeddings / node_embeddings.sum(axis=0, keepdims=True)
        entropy_embed = entropy(prob_embed + 1e-8, axis=0)
        components = [
            mean_embed, max_embed, min_embed, median_embed, range_embed,
            skewness_embed, kurtosis_embed, iqr_embed, entropy_embed
        ]

    # Replacing -inf, inf, and NaNs with 0 for consistency
        components = [np.nan_to_num(comp, nan=0.0, posinf=0.0, neginf=0.0) for comp in components]

        graph_embedding = np.concatenate(components)
        embeddings.append(graph_embedding)

    # Return 2D array num_graphs x dimensions
    return np.array(embeddings), np.array(labels)
