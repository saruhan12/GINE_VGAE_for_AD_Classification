# Alzheimer's Classification Pipeline (Alzheimers vs Healthy) with PET Images

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

### 1.2. Running the Pipeline for Graph Methods

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
