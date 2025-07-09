from bct import nbs_bct
import numpy as np
import networkx as nx

#Helper functions to convert our NetworkX graphs into nparray form and save the edge/node attributes for later use
def convert_nparr(graphs_AD, graphs_CN):
    X, Y = [], []
    X_node_attrs, Y_node_attrs = [], []
    X_edge_attrs, Y_edge_attrs = [], []

    for G in graphs_AD:
        X_node_attrs.append(nx.get_node_attributes(G, 'mean_activation'))
        edge_map = {(u,v): d.copy() for u,v,d in G.edges(data=True)}
        X_edge_attrs.append(edge_map)
        X.append(nx.to_numpy_array(G))
    X = np.transpose(np.array(X), (1,2,0))

    for G in graphs_CN:
        Y_node_attrs.append(nx.get_node_attributes(G, 'mean_activation'))
        edge_map = {(u,v): d.copy() for u,v,d in G.edges(data=True)}
        Y_edge_attrs.append(edge_map)
        Y.append(nx.to_numpy_array(G))
    Y = np.transpose(np.array(Y), (1,2,0))

    return X, Y, X_node_attrs, Y_node_attrs, X_edge_attrs, Y_edge_attrs

def convert_back(adj_matrices, node_attrs_list, edge_attrs_list, default_weight=1.0):
    converted_graphs = []
    for A, n_attrs, e_attrs in zip(adj_matrices, node_attrs_list, edge_attrs_list):
        G = nx.from_numpy_array(A)

        nx.set_node_attributes(G, n_attrs, 'mean_activation')

        fs_map = {frozenset({u, v}): d for (u, v), d in e_attrs.items()}

        for u, v, data in G.edges(data=True):
            key = frozenset({u, v})
            if key in fs_map:
                data.update(fs_map[key])
            data['weight'] = data.get('weight', default_weight)

        converted_graphs.append(G)
    return converted_graphs
##############################################
##Helper function to add the node/edge attributes back to our NetworkX graphs
def process_graphs(graphs_ad,graphs_cn, features_ad, features_cn):
    def process_graph(G, features):
        # Reindex nodes to 0..n-1
        mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)

        reordered_features = []
        for old_idx in mapping.keys():
            reordered_features.append(features[old_idx])
        reordered_features = np.array(reordered_features)

        for j in range(len(reordered_features)):
            G.nodes[j]["mean_activation"] = reordered_features[j]

        return G

    all_graphs = []

    for i in range(len(graphs_ad)):
        G = process_graph(graphs_ad[i], features_ad[i])
        all_graphs.append(G)

    for i in range(len(graphs_cn)):
        G = process_graph(graphs_cn[i], features_cn[i])
        all_graphs.append(G)


    return all_graphs

def nbs(graphs_AD, graphs_CN,features_ad, features_cn, labels,thresh=2.5 ):
    X, Y, X_attrs_n, Y_attrs_n, X_attrs_e, Y_attrs_e = convert_nparr(graphs_AD=graphs_AD, graphs_CN=graphs_CN)
    pvals, adj, null = nbs_bct(X, Y, thresh=thresh, k=1000, tail='both')

    X_select = []
    for k in range(X.shape[2]):
        graph = X[:, :, k]
        X_select.append(graph * adj)
    X_select = np.array(X_select)
    X_select = convert_back(X_select, X_attrs_n,X_attrs_e)  # Pass stored attributes

    Y_select = []
    for k in range(Y.shape[2]):
        graph = Y[:, :, k]
        Y_select.append(graph * adj)
    Y_select = np.array(Y_select)
    Y_select = convert_back(Y_select, Y_attrs_n,Y_attrs_e)
    selected = process_graphs(X_select, Y_select, features_ad, features_cn)
    return selected, labels


#BUGGY DO NOT TRY!
def nbs_with_features(graphs_ad, graphs_cn, features_ad, features_cn, labels, thresh=2.5):
    
    def stack_adj(graphs):
        return np.stack([nx.to_numpy_array(G) for G in graphs], axis=2)

    X = stack_adj(graphs_ad)   
    Y = stack_adj(graphs_cn)   
    pvals, adj, null = nbs_bct(X, Y, thresh=thresh, k=1000, tail='both')
    def apply_mask(stack):
        return stack * adj[..., None]

    Xm = apply_mask(X)
    Ym = apply_mask(Y)
    def rebuild(masked_stack, feats_list):
        out = []
        for i in range(masked_stack.shape[2]):
            A = masked_stack[:, :, i]
            G = nx.from_numpy_array(A)
            feat = feats_list[i]
            for n in G.nodes():
                G.nodes[n]['mean_activation'] = feat[n]
            out.append(G)
        return out

    ad_sel = rebuild(Xm, features_ad)
    cn_sel = rebuild(Ym, features_cn)

    return ad_sel + cn_sel, labels