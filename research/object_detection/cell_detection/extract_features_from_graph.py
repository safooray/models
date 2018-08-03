import itertools
import numpy as np
from inference_utils import load_frozen_graph 
from dpath_data_utils import get_file_name_from_serialized_example, get_labels_from_serialized_example, get_boxes_from_serialized_example_tensor 
from object_detection.core import standard_fields
from object_detection.inference import detection_inference
import os
import pickle
from PIL import Image
import tensorflow as tf


fields = standard_fields.DetectionResultFields


PATH_TO_GRAPH = "/home/yousefis/cell_seg/faster_bb_fc_final_2/inference_graph/frozen_inference_graph.pb"
OUT_PATH = "/mnt/Tardis/Yao/safoora/Data/ObjectDetectionGallery/Tissue/test_large/bbfc_model/gt_avg_features/"
#OUT_PATH = "/mnt/Tardis/Yao/safoora/Data/ObjectDetectionGallery/Tissue/test/no_finetune_model/gt_avg_features/"
TEST_REC_PATH = "/mnt/Tardis/Yao/safoora/Data/ObjectDetectionGallery/Tissue/test_large/test.record"
FEATS_TO_CROP_NAMES = list(map(lambda x:x+':0', [fields.block0_features, fields.block1_features,
                       fields.block2_features, fields.block3_features]))
#CROPPED_FEATS_NAMES = list(map(lambda x:x+':0', [fields.box_classifier_features])) 
CROPPED_FEATS_NAMES = [] 
USE_GT_BOXES = False
AVG_POOL = True

def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _save_as_numpy(feat_dict, global_pool_func, write_dir, **kwargs):
    mkdir_if_not_exists(write_dir)
    output_dict = {}
    output_dict.update({key:value for key, value in kwargs.items()})
    output_dict.update({key: global_pool_func(value, axis=(1, 2)) for key, value in feat_dict.items()})
    with open(os.path.join(write_dir, 'detection_features.pickle'), 'wb') as f:
        pickle.dump(output_dict, f)


def _normalize_to_0_255(feature):
    wmin = float(feature.min())
    wmax = float(feature.max())
    if (wmin == wmax):
        return feature
    feature *= (255.0/float(wmax-wmin))
    feature += abs(wmin)*(255.0/float(wmax-wmin))
    return feature


def _save_numpy_crop_as_image(crop, crop_dir, normalize=False):
    mkdir_if_not_exists(crop_dir)
    if normalize:
        crop = _normalize_to_0_255(crop)
    im = Image.fromarray(crop.astype(np.uint8))
    im.save(os.path.join(crop_dir, "crop_{}.png".format(j)))


def _save_numpy_crops_as_image(crops, crops_dir, normalize=False):
    for j in range(crops.shape[0]):
        crop = crops[j, ..., 0]
        _save_numpy_crop_as_image(crop, crops_dir,  normalize)


def extract_and_write_features(input_tfrecord_paths=[TEST_REC_PATH],
                               frozen_graph_path=PATH_TO_GRAPH,
                               write_dir=OUT_PATH,
                               cropped_features_tensor_names=CROPPED_FEATS_NAMES,
                               features_to_crop_tensor_names=FEATS_TO_CROP_NAMES):

    serialized_example_tensor, image_tensor = detection_inference.build_input(
        input_tfrecord_paths)

    g = load_frozen_graph(frozen_graph_path,
                          input_map={'image_tensor': image_tensor})
    # Get handle of tensors of interest in loaded graph.
    num_detections_tensor = tf.squeeze(
      g.get_tensor_by_name('num_detections:0'), 0)
    num_detections_tensor = tf.cast(num_detections_tensor, tf.int32)

    feats_to_crop_tensors_dict = {feature_tensor_name: g.get_tensor_by_name(feature_tensor_name)
                             for feature_tensor_name in features_to_crop_tensor_names}
    feats_to_crop_tensors_dict.update({'original_image': image_tensor})
    detected_scores_tensor = tf.squeeze(
      g.get_tensor_by_name('detection_scores:0'), 0)
    detected_scores_tensor = detected_scores_tensor[:num_detections_tensor]


    detected_boxes_tensor = tf.squeeze(
      g.get_tensor_by_name('detection_boxes:0'), 0)
    detected_boxes_tensor = detected_boxes_tensor[:num_detections_tensor]

    # Create condition to filter out tensors 
    # corresponding to objectness scores less than  0.50
    mask = detected_scores_tensor > 0.5
    mask.set_shape([None])

    # Apply the condition via boolean masks.
    cropped_feats_tensors_dict = {}
    for feature_tensor_name in cropped_features_tensor_names:
        cropped_feats_tensors_dict[feature_tensor_name] = tf.boolean_mask(tf.squeeze(
            g.get_tensor_by_name(feature_tensor_name), 0)[:num_detections_tensor], mask)

    # Use either ground truth or detected boxes for cropping features.
    # No need to apply boolean mask to ground truth boxes.
    if USE_GT_BOXES:
        boxes_tensor = get_boxes_from_serialized_example_tensor(serialized_example_tensor)
        num_boxes = tf.shape(boxes_tensor)[0]
    else:
        num_boxes = tf.reduce_sum(tf.cast(mask, tf.int32))
        # crop and resize 
        boxes_tensor = tf.boolean_mask(detected_boxes_tensor, mask)
    
    # Crop feature tensors using boxes.
    crop_tensors = {}
    for feature_to_crop_tensor_name, feature_to_crop_tensor in feats_to_crop_tensors_dict.items():
        crop_tensors[feature_to_crop_tensor_name] = tf.image.crop_and_resize(
                                     feature_to_crop_tensor,
                                     boxes_tensor,
                                     tf.zeros(num_boxes, dtype=tf.int32), 
                                     (16, 16)) 

    if AVG_POOL:
        global_pool_func = np.mean
    else:
        global_pool_func = np.max

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    with tf.Session(config=session_config) as sess:
        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners()
        try:
            for counter in itertools.count():
                tf.logging.log_every_n(tf.logging.INFO,
                                       'Processed %d images...', 10,
                                       counter)
                (serialized_example, 
                n, boxes, *feat_list) = tf.get_default_session().run(
                                          [serialized_example_tensor,
                                           num_boxes, boxes_tensor] +
                                           list(cropped_feats_tensors_dict.values())+
                                           list(crop_tensors.values()))
                
                filename = get_file_name_from_serialized_example(serialized_example)
                labels = get_labels_from_serialized_example(serialized_example)
                per_image_w_dir = os.path.join(write_dir, filename[:-len('.png')])
                feat_dict = dict(zip(list(cropped_feats_tensors_dict.keys())+
                                     list(crop_tensors.keys()),
                                     feat_list))
                if n > 0:
                    _save_as_numpy(feat_dict, global_pool_func, per_image_w_dir, boxes=boxes, labels=labels)
                    #_save_numpy_crops_as_image(feat_dict['original_image'], os.path.join(per_image_w_dir, 'crops'))
                
        except tf.errors.OutOfRangeError:
            tf.logging.info('Finished processing records')

if __name__=='__main__':
    extract_and_write_features()
