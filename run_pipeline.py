import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from src.utils import construct_graphs
import argparse
from src.utils import run_pipeline


def str2nonefloat_list(s: str):
    items = []
    for x in s.split(','):
        x = x.strip()
        if x.lower() == 'none':
            items.append(None)
        else:
            items.append(float(x))
    return items

def str2list(s: str):
    return [x.strip() for x in s.split(',') if x.strip()]

def main():
    parser = argparse.ArgumentParser(
        description= "AD/CN classification pipeline with feature select/embedding/classifier selection"
    )

    parser.add_argument(
        "--ad",
        type=str,
        required=True,
        help="Path to the folder containing AD PET Scans ib .nii format."
    )
    parser.add_argument(
        "--cn",
        type=str,
        required=True,
        help="Path to the folder containing Cn PET Scans ib .nii format"
    )
    parser.add_argument(
        "--atlas",
        type=str,
        required=True,
        help="Path to atlas image in .nii format."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="results_pipeline",
        help="Output directory for models, metrics, and summary CSV."
    )
    parser.add_argument(
        "--thresholds",
        type=str2nonefloat_list,
        default="None,0.98",
        help="Comma-separated list of thresholds to apply to edges in the pipeline, use 'None' for no threshold. (default: None,0.98)"
    )
    parser.add_argument(
        "--feature_selection_methods",
        type=str2list,
        default="None,NBS,DisparityFilter",
        help="Comma-separated list of feature selection methods. Options: None,NBS,DisparityFilter (default: None,NBS,DisparityFilter)"
    )
    parser.add_argument(
        "--embedding_methods",
        type=str2list,
        default="FeatherNode,VGAE_GINE",
        help="Comma-separated list of embedding methods. Options: FeatherNode,VGAE_GINE (default: FeatherNode,VGAE_GINE)"
    )
    parser.add_argument(
        "--classifiers",
        type=str2list,
        default="SVM,XGBoost",
        help="Comma-separated classifier names. Options:  SVM,XGBoost (default: SVM,XGBoost)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs for VGAE_GINE. (default: 100)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for VGAE_GINE training. (default: 32)"
    )
    args = parser.parse_args()

    alzh = args.ad


    healthy = args.cn
    atlas = args.atlas

    atlas_img = nib.load(atlas)
    masker_mean = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
    masker_std = NiftiLabelsMasker(labels_img=atlas_img, standardize=False,strategy="standard_deviation")
    
    #For ease of use and compatbility of edge selection methods, only the disparity filter graph construction was used, see utils for nbs specific construction.
    #With conversion tricks disparity filter compatible graphs also work with nbs edge selection reqs.
    ad,cn, features_ad, features_cn,labels = construct_graphs(alzh, healthy,masker_mean=masker_mean,masker_std=masker_std,atlas=atlas,select_method="disparity_filter")
    run_pipeline(ad,cn,labels, features_ad,features_cn,args.save_dir,args.thresholds,args.feature_selection_methods,args.embedding_methods,args.classifiers,args.epochs,args.batch_size)


if __name__ == "__main__":
    main()
