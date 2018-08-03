import io
import numpy as np
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import os
from PIL import Image
import tensorflow as tf


tf.flags.DEFINE_string('input_tfrecord_paths', None,
                       'A comma separated list of paths to input TFRecords.')
tf.flags.DEFINE_string('output_tfrecord_path', None,
                       'Path to the output TFRecord.')
tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights.')


def get_category_index(pipeline_config_path):
    """ 
    Get class label info from config file in case multiple classes are present.

    Arguments:
        pipeline_config_path: str. path to task config file.
        
    Returns:
        dict. Category index.
    """
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    input_config = configs['eval_input_config']

    label_map = label_map_util.load_labelmap(input_config.label_map_path)
    max_num_classes = max([item.id for item in label_map.item])

    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes)
    return label_map_util.create_category_index(categories)


def load_frozen_graph(path_to_graph, input_map=None):
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_graph, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='', input_map=input_map)
    return tf.get_default_graph() 


def run_inference_for_single_image(image, graph):    
    """
    Get references to model's output tensors, run inference in a tf session and
    return detection results in a dictionary.

    Arguments:
        image: numpy.ndarray. Input image.
        graph: tf.Graph(). Graph of the trained model.

    Returns: Dictionary of evaluated output tensors.
    """
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                
                
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})
            
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict	


def infer_and_visualize(img, detection_graph, category_index):
    """
    Call inference on image and visualize_results.

    Arguments:
        img: numpy.ndarray. Image in for of numpy array.
        gt_record: tf.record. Corresponding ground truth record.
        category_index: dict. Category index of classes.

    Returns: 
        numpy array of image overlaid with boxes(width, height, 3)
    """
    # Actual detection.
    output_dict = run_inference_for_single_image(img, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
      img,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      max_boxes_to_draw=None,
      min_score_thresh=.30,
      line_thickness=1,
      skip_labels=True,
      skip_scores=True)
    return img


def visualize_gt(img, gt_record, category_index={}):
    """
    Visualize ground truth boxes on the image.

    Arguments:
        img: numpy.ndarray. Image in for of numpy array.
        gt_record: tf.record. Corresponding ground truth record.
        category_index: dict. Category index of classes.
        
    Returns: 
        numpy array of image overlaid with boxes(width, height, 3)
    """
    # retrieve ground truth bounding boxes
    xmins = gt_record.features.feature['image/object/bbox/xmin'].float_list.value
    xmaxs = gt_record.features.feature['image/object/bbox/xmax'].float_list.value
    ymins = gt_record.features.feature['image/object/bbox/ymin'].float_list.value
    ymaxs = gt_record.features.feature['image/object/bbox/ymax'].float_list.value
    gt_boxes = np.array([ymins, xmins, ymaxs, xmaxs]).T
    # Visualization of theground truth.
    vis_util.visualize_boxes_and_labels_on_image_array(
          img,
          gt_boxes,
          None,
          None,
          category_index,
          instance_masks=None,
          use_normalized_coordinates=True,
          max_boxes_to_draw=1000,
          line_thickness=1,
          groundtruth_box_visualization_color='Gold',
          skip_labels=True,
          skip_scores=True)
    return img


#TODO(safoora): handle edge cases.
def infer_patch_by_patch(np_image, patch_size, detection_graph):
    """
    Given a large image, divide it into smaller patches, inder detections and
    visualize boxes on the each patch, then concatenate patches back together 
    for saving.
    Arguments:
        np_image: numpy array of shape (width, height, 3)
        patch_size: int. size of smaller square patches in pixels.
        detection_graph: tf.Graph(). Graph to use for detection.

    Returns: 
        numpy array of image overlaid with boxes(width, height, 3)
    """
    width, height, n_channels = np_image.shape
    
    n_patches_w = width // patch_size
    n_patches_h = height // patch_size

    img_row_list= []
    for j in range(n_patches_h):
        img_list = []
        for i in range(n_patches_w):
            img = np_image[j*patch_size:(j+1)*patch_size,
                           i*patch_size:(i+1)*patch_size, ...]
            infer_and_visualize(img, detection_graph, {})
            img_list.append(img)
        #Concat along x axis
        img_row_list.append(np.concatenate(img_list, axis=1))
    # Concat along y axis    
    return np.concatenate(img_row_list, axis=0)
