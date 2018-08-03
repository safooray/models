import csv 
import io
import numpy as np
from object_detection.core import standard_fields
from object_detection.utils import dataset_util
import os
import openslide as osd
import pickle
from PIL import Image
import sys
import tensorflow as tf

fields = standard_fields.DetectionResultFields
slim_example_decoder = tf.contrib.slim.tfexample_decoder

TILE_DIR = '/mnt/Tardis/Yao/safoora/Data/tiles'
IMS_URL = 'http://rpzmsvm0241.emea.roche.com/ims/'
PATCH_SIZE = 1000

sys.path.append('/home/yousefis/git/rhinoceros/rhinoceros')
sys.path.append('/home/yousefis/git/rhinoceros/rhinoceros/utils')
from api import IrisApi
import image_server_client


def get_project_ims_id_list(iris_project_id='1062915'):
    """ Get list of IMS ids of images in a Iris project.
    Arguments:
        iris_project_id: str. 

    Returns:
        ims_id_list: list of str. 
    """
    im_dict_dict, _ = IrisApi.get_iris_imlist(iris_project_id)
    ims_id_list = []
    for _, im_dict in im_dict_dict.items():
	    ims_id_list.append(im_dict['image_ims_id'])
    return ims_id_list


def get_all_tiles(ims_img_id, patch_size=PATCH_SIZE, ims_url=IMS_URL, tile_dir=TILE_DIR):
    """Get IMS image tile by tile.

    Arguments:
        ims_img_id: str.
        patch_size: int. height or width or square tiles in inches.
        ims_url: str. URL of the IMS server.
        tile_dir: str. Directory to save the tiles.
    """
    fields = IMS_image_fields()
    tiles_dir = os.path.join(tile_dir, str(ims_img_id))
    if not os.path.exists(tiles_dir):
        os.mkdir(tiles_dir)
    ims_client = image_server_client.IMSClient()
    image_atrs = ims_client.get_image_attributes(server_url=ims_url, image_id=ims_img_id)
    n_patches_w = image_atrs.width // patch_size
    n_patches_h = image_atrs.height // patch_size
    for j in range(n_patches_h):
        top = j * patch_size
        for i in range(n_patches_w):
            left = i * patch_size
            tile_bytes = ims_client.get_image(ims_url,
							  image_id=ims_img_id,
							  channel_or_zlayer_index=0,
							  resolution_index=0,
							  left=left, top=top,
							  width=patch_size,
							  height=patch_size)
            tile = Image.open(io.BytesIO(tile_bytes))
            tile.save(os.path.join(tiles_dir, '{}_{}.jpg'.format(i, j)))
    create_and_dump_info(tiles_dir, ims_id=ims_img_id, 
                         width=image_atrs.width, 
                         n_w_patches=n_patches_w, 
                         height=image_atrs.height, 
                         n_h_patches=n_patches_h)


def create_and_dump_info(write_path, **kwargs):
    """Makes a dictionary of kwargs and writes it in a pickle file in write_path.
    Use this e.g. to store information of the original image in the tiles folder.
    """
    info_dict = {}
    for key, value in kwargs.items():
	    info_dict[key] = value
    with open(os.path.join(write_path, 'info.pickle'), 'wb') as file:
        pickle.dump(info_dict, file)


def get_image_from_serialized_example(example):
    """Read image field from tf Example serialized as string.
    Arguments:
        example: str. TF serialized Example.

    Returns: 
        np.ndarray. RBG representation of the image.
        file_name: str. File name of the image.
    """
    record = tf.train.Example.FromString(example)
    img_string = record.features.feature['image/encoded'].bytes_list.value[0]
    file_name = record.features.feature['image/filename'].bytes_list.value[0].decode("utf8")
    image = Image.open(io.BytesIO(img_string))
    return np.array(image.convert('RGB')), file_name


def get_file_name_from_serialized_example(serialized_example):
    example = tf.train.Example.FromString(serialized_example)
    return example.features.feature['image/filename'].bytes_list.value[0].decode('utf8')

def get_labels_from_serialized_example(serialized_example):
    example = tf.train.Example.FromString(serialized_example)
    return list(example.features.feature['image/object/class/label'].int64_list.value)


def get_boxes_from_serialized_example_tensor(serialized_example_tensor):
    keys_to_features={
          standard_fields.TfExampleFields.object_bbox_xmin:
              tf.VarLenFeature(tf.float32),
          standard_fields.TfExampleFields.object_bbox_xmax:
              tf.VarLenFeature(tf.float32),
          standard_fields.TfExampleFields.object_bbox_ymin:
              tf.VarLenFeature(tf.float32),
          standard_fields.TfExampleFields.object_bbox_ymax:
              tf.VarLenFeature(tf.float32)
    }
    items_to_handlers = {
        standard_fields.InputDataFields.groundtruth_boxes: (
            slim_example_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'],
                                              'image/object/bbox/'))}
    decoder = slim_example_decoder.TFExampleDecoder(keys_to_features,
                                                    items_to_handlers)
    keys = decoder.list_items()
    tensors = decoder.decode(serialized_example_tensor, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    return tensor_dict[standard_fields.InputDataFields.groundtruth_boxes]


def image_to_binary(image_path):
    with tf.gfile.GFile(image_path, 'rb') as image_file:
        # Encoded image bytes
        return image_file.read() 


def tardis_win2unix_path(path):
    path = path.replace('\\', '/')
    path = path.replace('//sunnssvm04.sun.roche.com/', '/mnt/')
    return path.replace('tardis', 'Tardis')


def create_tf_example(image_path):
    filename = os.path.basename(image_path).encode('utf8')
    encoded_image_data = image_to_binary(image_path)
    
    slide = osd.open_slide(image_path)
    width, height = slide.dimensions
    
    image_format = b'png' 
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
    }))
    return tf_example


def create_gt_tf_example(image_path, ground_truth_dict, classes_text, classes_id):
    filename =  os.path.basename(image_path) # extarct image filename from the path.

    encoded_image_data = image_to_binary(image_path)
    
    slide = osd.open_slide(image_path)
    width, height = slide.dimensions
    
    image_format = b'png' 
    xmins = [xmin/width for xmin in ground_truth_dict['xmins']] # List of normalized left xs (1 per box)
    xmaxs = [xmax/width for xmax in ground_truth_dict['xmaxs']] # List of normalized right xs (1 per box)
    ymins = [ymin/height for ymin in ground_truth_dict['ymins']] # List of normalized top ys (1 per box)
    ymaxs = [ymax/height for ymax in ground_truth_dict['ymaxs']] # List of normalized bottom ys (1 per box)

    assert all([xmin <= 1 for xmin in xmaxs]) # test the normalization.
        
    classes_text = [c.encode('utf8') for c in classes_text]
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes_id),
    }))
    return tf_example


def csv_boxes_to_tfrecord(path_to_csv_boxes, out_path, class_map, csv_info):
    n_classes = len(class_map)
    with open(path_to_csv_boxes) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        writer = tf.python_io.TFRecordWriter(out_path)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            xmins = []
            ymins = []
            xmaxs = []
            ymaxs = []
            classes_text = []
            classes_ids = []
            image_path = line[0]
            dim = 128
            cur = csv_info['last_non_box_col']
            for c in range(n_classes):
                box_count = int(line[csv_info['file_name_col'] + c])
                for j in range(box_count):
                    xmin = int(line[j + cur])
                    xmins.append(xmin)
                    ymin = int(line[j + box_count + cur])
                    ymins.append(ymin)
                    xmaxs.append(int(line[j + 2*box_count + cur]) + xmin - 1)
                    ymaxs.append(int(line[j + 3*box_count + cur]) + ymin - 1)
                    classes_ids.append(c+1)
                    classes_text.append(class_map[c+1])
                cur = cur + 4 * box_count
                while cur < len(line) and len(line[cur]) == 0:
                    cur = cur + 1
                    
            ground_truth = {'xmins':xmins, 'ymins':ymins, 'xmaxs':xmaxs, 'ymaxs':ymaxs}
            tf_example = create_gt_tf_example(tardis_win2unix_path(image_path), ground_truth, classes_text, classes_ids)        
            
            writer.write(tf_example.SerializeToString())
                
        writer.close()   

def csv_boxes_to_tfrecord_2(path_to_csv_boxes, out_path, class_map, csv_info):
    n_classes = len(class_map)
    with open(path_to_csv_boxes) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        writer = tf.python_io.TFRecordWriter(out_path)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            xmins = []
            ymins = []
            xmaxs = []
            ymaxs = []
            classes_text = []
            classes_ids = []
            image_path = line[0]
            dim = 128
            cur = csv_info['last_non_box_col']
            for c in range(n_classes):
                box_count = int(line[csv_info['file_name_col'] + c])
                for j in range(box_count):
                    xmin = int(line[j + cur])
                    xmins.append(xmin)
                    ymin = int(line[j + box_count + cur])
                    ymins.append(ymin)
                    xmaxs.append(int(line[j + 2*box_count + cur]) + xmin - 1)
                    ymaxs.append(int(line[j + 3*box_count + cur]) + ymin - 1)
                    classes_ids.append(1)
                    classes_text.append("Cell")
                cur = cur + 4 * box_count
                while cur < len(line) and len(line[cur]) == 0:
                    cur = cur + 1
                    
            ground_truth = {'xmins':xmins, 'ymins':ymins, 'xmaxs':xmaxs, 'ymaxs':ymaxs}
            tf_example = create_gt_tf_example(tardis_win2unix_path(image_path), ground_truth, classes_text, classes_ids)        
            
            writer.write(tf_example.SerializeToString())
                
        writer.close()   
