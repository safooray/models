import itertools
import numpy as np
from object_detection.utils import visualization_utils as vis
from object_detection.core import standard_fields
from dpath_data_utils import get_image_from_serialized_example
import os
import pandas as pd
import pickle
import PIL
from sklearn.cluster import DBSCAN, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

FEAT_DIR = '/mnt/Tardis/Yao/safoora/Data/ObjectDetectionGallery/Tissue/test/comb_model/gt_avg_features/'
#FEAT_DIR = '/mnt/Tardis/Yao/safoora/Data/ObjectDetectionGallery/Tissue/test/no_finetune_model/gt_avg_features/'
OUT_DIR = '/mnt/Tardis/Yao/safoora/Data/ObjectDetectionGallery/Tissue/test/comb_model/gt_avg_cluster_analysis_2'
#OUT_DIR = '/mnt/Tardis/Yao/safoora/Data/ObjectDetectionGallery/Tissue/test/no_finetune_model/gt_avg_cluster_analysis/'
#REC_PATH = '/mnt/Tardis/Yao/safoora/Data/ObjectDetectionGallery/Tissue/test/test.record'


fields = standard_fields.DetectionResultFields
VIS_CLUSTERS = False
EVAL_CLUSTERS = True


def visualize_clustered_boxes_on_image(image, boxes, cluster_labels):
    """
    Calls the TF OD visualization function with the right parameters for the 
    purpose of visualizing boxes on image colored by cluster label.

    Arguments:
        image: numpy.ndarray. 3D flaot image array.
        boxes: numpy.ndarray. 2D float array of shape (n_boxes, 4).
        cluster_labels: 1D array of shape (n_boxes,) denoting box icluster labels.

    Returns:
        3D float image array with color annotated  boxes overlaid. 

    """
    return vis.visualize_boxes_and_labels_on_image_array(image,
                                                  boxes,
                                                  cluster_labels, 
                                                  np.ones(cluster_labels.shape[0]), 
                                                  {},
                                                  use_normalized_coordinates=True,
                                                  max_boxes_to_draw=None,
                                                  skip_scores=True, 
                                                  skip_labels=True)


def read_pickle(read_path):
    """
    Reads pickle file and returns the object inside.

    Arguments:
        read_path: str. path to the saved pickle file.

    Returns:
        The object read from the pickle file.
    """
    with open(read_path, 'rb') as f:
        in_dict = pickle.load(f)
    return in_dict


def get_clustering_info(feature_name, read_dir=FEAT_DIR):
    image_cell_info = {}
    files_list = sorted(os.listdir(read_dir))
    X_list = []
    y_list = []
    start = 0
    # Iterates over all model output files
    for filename in files_list:
        # Reads output file.
        detection_dict = read_pickle(os.path.join(read_dir, filename, 'detection_features.pickle'))
        assert feature_name in detection_dict.keys() 
        assert 'labels' in detection_dict.keys() 
        # Discards boxes with score lower than threshold.
        #detection_dict = apply_score_threshold(detection_dict)
        
        # Puts selected regions of all images together for clustering purposes,
        # while keeping track of where the regions corresponding to each image
        # end up in the concatenation result.
        end = start + detection_dict[feature_name].shape[0]
        image_cell_info.update({filename: detection_dict})
        image_cell_info[filename].update({'start':start, 'end':end})
        start = end
        
        X_list.append(detection_dict[feature_name])
        y_list.append(detection_dict['labels'])
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(y_list, axis=0)
    return X, Y, image_cell_info


def apply_score_threshold(detection_dict, thresh=0.5):
    """
    Selects values from the dict that correspond to box scores > thresh.
    Arguments:
        detection_dict: dict. A dictionary contataining model outputs.
        It is the return value of the read_model_outputs function.

    Returns:
        Dictionary with same keys as input dictionary but values that 
        correspond to box scores > thresh.
    """
    cond = (detection_dict['detected_scores'] > thresh)
    return {key:detection_dict[key][cond] for key in detection_dict.keys()}


def accuracy_score(labels, preds):
    """
    Calculates the fraction of samples that are minority in their predicted clusters.
    Arguments:
        labels: np.ndarray of shape (n_samples,). Ground truth cluster labels.
        preds: np.ndarray of shape (n_samples,). Predicted clusters.
    """
    contingency_matrix = metrics.cluster.supervised.contingency_matrix(labels, preds)
    print(contingency_matrix)
    n_samples = len(labels)
    max_out = contingency_matrix
    max_out[max_out.argmax(0), np.arange(max_out.shape[1])] = 0
    
    return 1 - np.sum(max_out)/n_samples


def mkdir_if_not_exits(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__=='__main__':
    
    #feature_names = [fields.block3_features+':0',fields.block2_features+':0',fields.block1_features+':0', fields.block0_features+':0', 'original_image']
    feature_names = [fields.block0_features+':0']
    Ks = [5]
    df_columns = ['feature', 'n_clusters', 'homogeneity', 'accuracy', 'mutual_info', 'method']
    df_columns = ['feature', 'n_clusters','accuracy', 'method']
    df = pd.DataFrame(columns=df_columns)
    for feature_name in feature_names: 
        X, Y, image_cell_info = get_clustering_info(feature_name, read_dir=FEAT_DIR)
        # Standardizes data.
        X = StandardScaler().fit_transform(X)
        # Performs clustering.
        for K in Ks:
            preds = AgglomerativeClustering(n_clusters=K).fit_predict(X)
            if EVAL_CLUSTERS:
                df_data = [feature_name, K,
                           accuracy_score(Y, preds), 
                           'hierarchical']
                df = df.append(dict(zip(df_columns, df_data)), ignore_index=True)

            # Reads images from TF records and visualizes boxes on them.
            if VIS_CLUSTERS:
                for example in tf.python_io.tf_record_iterator(REC_PATH):
                    np_image, filename = get_image_from_serialized_example(example)
                    try:
                        img_info_dict = image_cell_info[filename[:-4]]
                    except KeyError:
                        print("No features were found for image {}.".format(filename))
                        continue
                    img_box_labels = preds[img_info_dict['start']: img_info_dict['end']]
                    img_box_gt = Y[img_info_dict['start']: img_info_dict['end']]
                    annotated_img = visualize_clustered_boxes_on_image(np_image,
                                                             img_info_dict['boxes'],
                                                             img_box_labels)
                    cl_dir = os.path.join(OUT_DIR, 'k_{}_{}'.format(K, feature_name), filename)
                    mkdir_if_not_exits(cl_dir)
                    PIL.Image.fromarray(annotated_img).save(os.path.join(cl_dir, 'clustering.png'))                                
                    gt_annotated_img = visualize_clustered_boxes_on_image(np_image,
                                                             img_info_dict['boxes'],
                                                             img_box_gt) 
                    PIL.Image.fromarray(gt_annotated_img).save(os.path.join(cl_dir, 'ground_truth.png'))                                
    if EVAL_CLUSTERS:
        mkdir_if_not_exits(OUT_DIR)
        with open(os.path.join(OUT_DIR, 'results.pick'), 'wb') as f:
            pickle.dump(df, f)
