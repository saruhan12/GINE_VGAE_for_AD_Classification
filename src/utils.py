import os
import numpy as np
import networkx as nx
import nibabel as nib
import networkx as nx
import torch
from nilearn.input_data import NiftiLabelsMasker
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd
import matplotlib.pyplot as plt
from src.classifiers import train_SVM,train_XGBoost
from src.vgae import get_graph_embeddings_vgae
from src.filters.dis_filt import disparity_f_on_all
from src.filters.nbs import nbs
from src.feather import calcul_graph_embed_feather
from src.filters.nbs import process_graphs

#node features -> (mean_of_roi, std_of_roi)
#edge features -> connectivity between 2 rois(specific equation shown below)
def construct_graph_for_disparity_filter(img_data, masker_mean, masker_std, labels):
    G = nx.Graph()

    voxel_ts = masker_mean.transform(img_data)
    mean_values = voxel_ts.mean(axis=0)
    standard_dev = masker_std.transform(img_data)

    node_features = []

    for i in range(len(mean_values)):
        G.add_node(i, label=f"{labels[i]}")
        node_features.append(np.array([mean_values[i], standard_dev[0][i]]))

        for j in range(len(mean_values)):
            #Connectivity equation 
            distance = -np.sqrt(np.sum((np.array([mean_values[i], standard_dev[0][i]]) - np.array([mean_values[j], standard_dev[0][j]])) ** 2))
            weight = np.exp(distance)
            G.add_edge(i, j, weight=weight)

    node_features = np.array(node_features)

    return G, node_features

def costruct_graph_for_NBS(img_data, masker_mean, masker_std,atlas):
    G = nx.Graph()
    voxel_ts = masker_mean.transform(img_data)
    mean_values = voxel_ts.mean(axis=0)
    standard_dev = masker_std.transform(img_data)

    for i in range(len(mean_values)):
        G.add_node(i,mean_activation=(mean_values[i],standard_dev[0][i]))
        for j in range(0, len(mean_values)):
            distance = -np.sqrt((np.sum(np.square(np.array(mean_values[i],standard_dev[0][i]) - np.array(mean_values[j],standard_dev[0][j])))))
            weight = np.exp(distance)
            G.add_edge(i, j, weight=weight)

    return G

#Helper funciton to apply an edge weight threshold to NetworkX graphs, our 0.98 threshold is a heauristic approach
def apply_edge_weight_threshold(G, weight_threshold=0.98):
    G_filtered = G.copy()  # To avoid modifying the original graph
    edges_to_remove = [(u, v) for u, v, d in G_filtered.edges(data=True) if d.get('weight', 0) < weight_threshold]
    G_filtered.remove_edges_from(edges_to_remove)
    return G_filtered

def apply_threshold(graphs, threshold):
  filtered = []
  for graph in graphs:
    filtered.append(apply_edge_weight_threshold(graph, threshold))
  return filtered



def construct_graphs(ad_fold, cn_fold, masker_mean, masker_std,atlas,select_method):

    #Given 2 folders(AD and CN) the method constructs the NetworkX grpahs according to the selected method(disparity_filter or nbs)
    #atlas is the brain atlas in .nii format
    #masker_mean and masker_std are the nparrays of 1x120 which contain the mean values and standart derivation of each ROI, extracter using the nilearn library
    #see run_pipeline.py for example usage

    print("construct nx graphs")
    masker_mean.fit()
    masker_std.fit()
    ad_g = []
    cn_g = []
    y = []
    labels_img = nib.load(atlas)
    labels_data = labels_img.get_fdata()
    unique_labels = np.unique(labels_data)[1:]
    if select_method ==  "disparity_filter":
        features_ad = []
        features_cn = []
        for patient_file in os.listdir(ad_fold):
            img = nib.load(os.path.join(ad_fold, patient_file))
            G, feature = construct_graph_for_disparity_filter(img, masker_mean,masker_std,unique_labels)
            features_ad.append(feature)
            ad_g.append(G)
            y.append(1)  

        for healthy_file in os.listdir(cn_fold):
            img = nib.load(os.path.join(cn_fold, healthy_file))
            G, feature = construct_graph_for_disparity_filter(img, masker_mean,masker_std,unique_labels)
            features_cn.append(feature)
            cn_g.append(G)
            y.append(0)  

        y = np.array(y)
        return ad_g, cn_g,features_ad,features_cn, y
    elif select_method ==  "nbs":
        for patient_file in os.listdir(ad_fold):
            img = nib.load(os.path.join(ad_fold, patient_file))
            G = costruct_graph_for_NBS(img, masker_mean,masker_std,atlas)
            ad_g.append(G)
            y.append(1)  

        for healthy_file in os.listdir(cn_fold):
            img = nib.load(os.path.join(cn_fold, healthy_file))
            G = costruct_graph_for_NBS(img, masker_mean,masker_std,atlas)
            cn_g.append(G)
            y.append(0)  
        y = np.array(y)

        return ad_g, cn_g, y




def evaluate_classification(y_true, y_pred, y_prob=None):
    results = {}
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['f1_score'] = f1_score(y_true, y_pred, average='weighted')
    results['recall'] = recall_score(y_true, y_pred, average='weighted')
    results['precision'] = precision_score(y_true, y_pred, average='weighted')

    return results


def filter_empty_and_edgeless(graphs, labels):
    """
    Drop any graph with 0 nodes or 0 edges, and keep labels in sync.
    """
    filtered = [
        (G, lbl)
        for G, lbl in zip(graphs, labels)
        if (G.number_of_nodes() > 0 and G.number_of_edges() > 0)
    ]
    dropped = len(graphs) - len(filtered)
    if dropped:
        print(f"⚠️ Dropping {dropped} empty/edgeless graph(s) before embedding")
    # unzip back into two lists (or tuples)
    if filtered:
        Gs, Ls = zip(*filtered)
        return list(Gs), list(Ls)
    else:
        return [], []


def run_pipeline(ad_graphs,cn_graphs,labels,features_ad,features_cn,
                 save_dir='results_pipeline',
                 thresholds=[None, 0.98],
                 feature_selection_methods=[None, 'NBS', 'DisparityFilter'],
                 embedding_methods=['FeatherNode', 'VGAE_GINE'],
                 classifiers=['SVM', 'XGBoost'],
                 epochs=100,
                 batch_size=32):
    
    os.makedirs(save_dir, exist_ok=True)
    results_records = []

    # Iterate all combinations
    for threshold in tqdm(thresholds, desc='Thresholds'):
        # Apply thresholding if needed 
        if threshold is not None:
            # Apply threshold filter on graphs
            filtered_ad= apply_threshold(ad_graphs, threshold)
            filtered_cn= apply_threshold(cn_graphs, threshold)
        else:
            filtered_ad = ad_graphs
            filtered_cn = cn_graphs

        for fs_method in tqdm(feature_selection_methods, desc='Feature Selection', leave=False):
            if fs_method == 'NBS':
                selected_graphs, selected_labels = nbs(filtered_ad, filtered_cn, features_ad=features_ad, features_cn=features_cn,labels=labels)
            elif fs_method == 'DisparityFilter':
                selected_graphs, selected_labels = disparity_f_on_all(filtered_ad, filtered_cn, features_ad=features_ad, features_cn=features_cn,labels=labels)
                selected_graphs, selected_labels = filter_empty_and_edgeless(selected_graphs, selected_labels)
            else:
                selected_graphs = process_graphs(filtered_ad,filtered_cn,features_ad=features_ad,features_cn=features_cn)
                selected_labels = labels

            for embedding_method in tqdm(embedding_methods, desc='Embedding', leave=False):
                if embedding_method == 'FeatherNode':
                    embeddings, emb_labels = calcul_graph_embed_feather(selected_graphs, selected_labels)
                elif embedding_method == 'VGAE_GINE':
                    embeddings, emb_labels = get_graph_embeddings_vgae(selected_graphs, selected_labels, epochs=epochs, batch_size=batch_size,pretrained_weights_path="/content/drive/MyDrive/KG_project/gine_vgae_cora.pth")
                else:
                    raise ValueError(f"Unknown embedding method: {embedding_method}")

                for clf_name in tqdm(classifiers, desc='Classifier', leave=False):
                    if clf_name == 'SVM':
                        clf_model,y_test, y_pred, y_prob = train_SVM(embeddings, emb_labels,)
                    elif clf_name == 'XGBoost':
                        clf_model,y_test, y_pred, y_prob = train_XGBoost(embeddings, emb_labels)
                    else:
                        raise ValueError(f"Unknown classifier: {clf_name}")

                    # Evaluate
                    metrics = evaluate_classification(y_test, y_pred, y_prob)

                    # Save results
                    combo_name = f"thr_{threshold}_fs_{fs_method}_emb_{embedding_method}_clf_{clf_name}"
                    combo_dir = os.path.join(save_dir, combo_name)
                    os.makedirs(combo_dir, exist_ok=True)

                    # Save model
                    torch.save(clf_model, os.path.join(combo_dir, 'classifier.pt'))

                    # Save embeddings and labels
                    np.save(os.path.join(combo_dir, 'embeddings.npy'), embeddings)
                    np.save(os.path.join(combo_dir, 'labels.npy'), emb_labels)

                    # Save metrics to CSV
                    metrics_df = pd.DataFrame([metrics])
                    metrics_df.to_csv(os.path.join(combo_dir, 'evaluation.csv'), index=False)

                    # Record for overall summary
                    results_records.append({
                        'threshold': threshold,
                        'feature_selection': fs_method,
                        'embedding': embedding_method,
                        'classifier': clf_name,
                        **metrics
                    })

    # Save summary results for all combinations
    summary_df = pd.DataFrame(results_records)
    summary_df.to_csv(os.path.join(save_dir, 'summary_results.csv'), index=False)
    print(f"Pipeline finished. Summary saved to {os.path.join(save_dir, 'summary_results.csv')}")