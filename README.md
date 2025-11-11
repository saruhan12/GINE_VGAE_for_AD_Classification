# Alzheimer's Classification Pipeline (Alzheimers vs Healthy) with PET Images
This is the source code for the following paper: 
[S. M. Gürbüz and M. Adel, "An Edge Feature Inclusive Variational Graph Autoencoder for Pet-Driven Alzheimer's Diagnosis," 2025 Fourteenth International Conference on Image Processing, Theory, Tools & Applications (IPTA), Istanbul, Turkiye, 2025, pp. 1-4 doi: 10.1109/IPTA66025.2025.11222021](https://ieeexplore.ieee.org/document/11222021).
## 1. Graph Methods Description

This pipeline takes two brain connectivity graph datasets (Alzheimer’s and Healthy groups), which are creted from 18F-FDG PET scans, as input and performs the following steps:

1. **Connectivity Threshold**
   • Filtering of edges based on a weight threshold.
2. **Feature Selection**
   • Disparity Filter(from https://github.com/DerwenAI/disparity_filter)
   • Network-Based Statistic (NBS) (from https://github.com/aestrivex/bctpy)
3. **Graph Re-embedding**
   • FeatherNode 
   • VGAE-GINE 
4. **Classification**
   • SVM and XGBoost with hyperparameter search and nested cross-validation

Output of the pipeline includes:

* Performance reports (F1, accuracy, classification report)
* Best hyperparameter sets
* (Optional) Stored embeddings for each graph
## 1.2 Our method and the results

Our proposed **GINE–VGAE** is a novel graph embedding model designed for PET-based Alzheimer’s diagnosis.  
It extends the classical Variational Graph Autoencoder (VGAE) by integrating **GINEConv layers**, which account for both **node features** and **continuous edge weights**—allowing the model to capture rich metabolic connectivity patterns between brain regions.

The pipeline:
1. Constructs **subject-specific PET connectivity graphs** using anatomical ROIs and Network-Based Statistic (NBS) edge pruning.  
2. Learns **latent graph embeddings** via GINE–VGAE, where GINEConv layers enable edge-aware message passing.  
3. Applies **statistical pooling** (mean, variance, entropy, etc.) to obtain compact graph-level vectors.  
4. Classifies subjects using **SVM** or **XGBoost** with nested cross-validation.

This approach bridges the gap between handcrafted graph features and purely unsupervised embeddings, achieving **93.8% accuracy** and an **F1-score of 0.937** on the ADNI PET dataset—outperforming other methods below.

| Method (Ref.)           | Modality        | Embedding               | Classifier      | Acc. %  |
|--------------------------|-----------------|--------------------------|-----------------|---------|
| Node2Vec + SVM [13]      | fMRI            | Node2Vec                 | SVM             | 90.6    |
| MG2G VGAE + XGBoost [14] | fMRI            | Multi-Gaussian VGAE      | XGBoost         | 89–90   |
| Multimodal GCN [28]      | PET + Clinical  | Population GCN           | GCN             | 90.4    |
| Graph Kernel SVM [17]    | MRI             | HGK-SP Kernel            | SVM             | 83.8    |
| **Proposed method**      | PET             | **GINE–VGAE**            | **SVM/XGBoost** | **93.8**|
| **Baseline**             | PET             | None                     | SVM/            | 91      |

### 1.3. Running the Pipeline for Graph Methods

The main script is located at `methode_graph/run_pipeline.py`. To run the pipeline, execute:

```bash
python run_pipeline.py \
  --ad_dir PATH/TO/AD_GRAPHS \
  --hc_dir PATH/TO/HEALTHY_GRAPHS \
  --features_ad PATH/TO/AD_FEATURES.npy \
  --features_hc PATH/TO/HEALTHY_FEATURES.npy \
  --labels PATH/TO/LABELS.npy \
  --thresholds 0.98 0.95 \
  --fs_methods DisparityFilter NBS \
  --emb_methods FeatherNode VGAE_GINE \
  --classifiers svm xgb \
  --epochs 100 \
  --batch_size 32 \
  --output_dir results/
```
### References
[13]R. K. Lama and G.-R. Kwon, “Diagnosis of alzheimer’s disease using
brain network,” Frontiers in Neuroscience, vol. 15, p. 605115, 2021.

[14]J. Xu, L. Zhang, M. Chen, R. Wang, and H. Zhao, “Multiple graph
gaussian embedding for alzheimer’s disease classification,” IEEE Trans-
actions on Medical Imaging, vol. 39, no. 12, pp. 4123–4133, 2020.

[17]L. J. C. de Mendonc¸a and R. J. Ferrari, “Alzheimer’s disease classifica-
tion based on graph kernel svms constructed with 3d texture features,”
Expert Systems with Applications, vol. 211, p. 118633, 2023.

[28]G.-B. Lee, Y.-J. Jeong, D.-Y. Kang, H.-J. Yun, and M. Yoon,
“Multimodal feature fusion-based graph convolutional networks for
alzheimer’s disease stage classification using f-18 florbetaben brain pet
images and clinical indicators,” PLOS ONE, vol. 19, no. 12, pp. 1–23, 12
2024. Available: https://doi.org/10.1371/journal.pone.0315809
