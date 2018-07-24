import itertools
import numpy as np
from object_detection.utils import visualization_utils as vis
from object_detection.core import standard_fields
from dpath_data_utils import read_image_from_tfrecord
import os
import pandas as pd
import pickle
import PIL
from sklearn.cluster import DBSCAN, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

FEAT_DIR = '/mnt/Tardis/Yao/safoora/Data/ObjectDetectionGallery/Tissue/test/features/'
OUT_DIR = '/mnt/Tardis/Yao/safoora/Data/ObjectDetectionGallery/Tissue/test/gt_ft_cluster_analysis'
REC_PATH = '/mnt/Tardis/Yao/safoora/Data/ObjectDetectionGallery/Tissue/test/test.record'

fields = standard_fields.DetectionResultFields
VIS_CLUSTERS = False

def visualize_clustered_boxes_on_image(image, boxes, cluster_labels):
    """
    Calls the TF OD visualization function with the right parameters for the 
    purpose of visualizing boxes on image colored by cluster label.
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


def read_model_output(read_path):
    """
    Reads model outputs from pickle file and returns them.

    Arguments:
        read_path: str. path to the saved outputs pickle file.

    Returns:
        The dictionary read from the pickle file.
    """
    with open(read_path, 'rb') as f:
        in_dict = pickle.load(f)
    return in_dict


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


if __name__=='__main__':
    
    #feature_name = 'original_image'
    feature_names = [fields.block3_features+':0',fields.block2_features+':0',fields.block1_features+':0', fields.block0_features+':0', 'original_image']
    #feature_names = [fields.block0_features+':0']
    Ks = [2, 3, 5, 10, 20, 25, 30]
    df_columns = ['feature', 'n_clusters', 'homogegeity', 'method']
    df = pd.DataFrame(columns=df_columns)
    for feature_name in feature_names: 
        files_list = os.listdir(FEAT_DIR)
        X_list = []
        y_list = []
        n_samples = {}
        cluster_info = {}
        start = 0
        # Iterates over all model output files
        for filename in files_list:
            # Reads output file.
            detection_dict = read_model_output(os.path.join(FEAT_DIR, filename, 'detection_features.pickle'))
            # Discards boxes with score lower than threshold.
            #detection_dict = apply_score_threshold(detection_dict)
            
            # Puts selected regions of all images together for clustering purposes,
            # while keeping track of where the regions corresponding to each image
            # end up in the concatenation result.
            end = start + detection_dict[feature_name].shape[0]
            cluster_info.update({filename: detection_dict})
            cluster_info[filename].update({'start':start, 'end':end})
            start = end
            
            X_list.append(detection_dict[feature_name])
            y_list.append(detection_dict['labels'])
        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(y_list, axis=0)
        # Standardizes data.
        X = StandardScaler().fit_transform(X)
        # Performs clustering.
        for K in Ks:
            labels = AgglomerativeClustering(n_clusters=K).fit_predict(X)
            df_data = [feature_name, K, metrics.homogeneity_score(Y, labels), 'hierarchical']
            df = df.append(dict(zip(df_columns, df_data)), ignore_index=True)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    with open(os.path.join(OUT_DIR, 'results.pick'), 'wb') as f:
        pickle.dump(df, f)

    # Reads images from TF records and visualizes boxes on them.
    if VIS_CLUSTERS:
        for example in tf.python_io.tf_record_iterator(REC_PATH):
            np_image, filename = read_image_from_tfrecord(example)
            try:
                img_info_dict = cluster_info[filename[:-4]]
            except KeyError:
                continue
            img_box_labels = labels[img_info_dict['start']: img_info_dict['end']]
            annotated_img = visualize_clustered_boxes_on_image(np_image,
                                                     img_info_dict['boxes'],
                                                     img_box_labels)
            PIL.Image.fromarray(annotated_img).save(os.path.join(OUT_DIR, filename))                                
