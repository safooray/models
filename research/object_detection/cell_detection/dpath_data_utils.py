#from api import IrisApi
#import image_server_client
import io
import numpy as np
import os
import pickle
from PIL import Image
import tensorflow as tf

TILE_DIR = '/mnt/Tardis/Yao/safoora/Data/tiles'
IMS_URL = 'http://rpzmsvm0241.emea.roche.com/ims/'
PATCH_SIZE = 1000

"""
def get_project_ims_id_list(iris_project_id='1062915'):
	im_dict_dict, _ = IrisApi.get_iris_imlist(iris_project_id)
	iris_id_list = []
	ims_id_list = []
	for _, im_dict in im_dict_dict.items():
		iris_id_list.append(im_dict['nid'])
		ims_id_list.append(im_dict['image_ims_id'])
		return ims_id_list


def get_all_tiles(ims_img_id):
	tiles_dir = os.path.join(TILE_DIR, str(ims_img_id))
	if not os.path.exists(tiles_dir):
		os.mkdir(tiles_dir)
	ims_client = image_server_client.IMSClient()
	image_atrs = ims_client.get_image_attributes(server_url=IMS_URL, image_id=ims_img_id)
	n_patches_w = image_atrs.width // PATCH_SIZE
	n_patches_h = image_atrs.height // PATCH_SIZE
	print(n_patches_h, n_patches_w)
	for j in range(n_patches_h):
		break
		top = j * PATCH_SIZE
		for i in range(n_patches_w):
			break
			left = i * PATCH_SIZE
			print('now at:{} {}'.format(top, left))
			tile_bytes = ims_client.get_image(IMS_URL,
							  image_id=ims_img_id,
							  channel_or_zlayer_index=0,
							  resolution_index=0,
							  left=left, top=top,
							  width=PATCH_SIZE,
							  height=PATCH_SIZE)

			tile = Image.open(io.BytesIO(tile_bytes))
			tile.save(os.path.join(tiles_dir, '{}_{}.jpg'.format(i, j)))
	create_and_dump_info(image_atrs.width, n_patches_w, image_atrs.height, n_patches_h, tiles_dir)
"""

def read_image_from_tfrecord(example):
    record = tf.train.Example.FromString(example)
    img_string = record.features.feature['image/encoded'].bytes_list.value[0]
    file_name = record.features.feature['image/filename'].bytes_list.value[0].decode("utf8")
    image = Image.open(io.BytesIO(img_string))
    return np.array(image.convert('RGB')), file_name


def create_and_dump_info(w, n_w, h, n_h, path):
	info = {}
	info['width'] = w
	info['n_w'] = n_w
	info['height'] = h
	info['n_h'] = n_h
	with open(os.path.join(path, 'info.pickle'), 'wb') as file:
		pickle.dump(info, file)
