import unittest
import numpy as np
import os
import pickle
from feature_clustering import accuracy_score, get_clustering_info
TEST_DIR = '/home/yousefis/.conda/envs/safoora_cell/lib/python3.6/site-packages/tensorflow/models/research/object_detection/cell_detection/test'

class TestClusteringMethods(unittest.TestCase):
    def setUp(self):
        self.n_cells = 100
        self.image_1_n_cells = 40
        self.image_2_n_cells = 60
        self.featsize = 1024
        self.labels = np.concatenate([np.zeros((70,)), np.ones((30,))])
        self.preds = np.concatenate([np.zeros((60,)), np.ones((15,)), np.ones((25,))*2])


    def test_accuracy_score(self):
        actual = accuracy_score(self.labels, self.preds)
        expected = 0.95
        self.assertEqual(expected, actual)


    def test_get_clustering_info(self):
        image_1_feats = np.random.rand(self.image_1_n_cells, self.featsize)  
        output_dict = {'feat': image_1_feats, 'labels': np.ones((self.image_1_n_cells))}
        
        with open(os.path.join(TEST_DIR, 'image_1', 'detection_features.pickle'), 'wb') as f:
            pickle.dump(output_dict, f)

    
        image_2_feats = np.random.rand(self.image_2_n_cells, self.featsize)  
        output_dict = {'feat': image_2_feats, 'labels': np.ones((self.image_2_n_cells))}
        
        with open(os.path.join(TEST_DIR, 'image_2', 'detection_features.pickle'), 'wb') as f:
            pickle.dump(output_dict, f)
        
        X, Y, image_cell_info = get_clustering_info(TEST_DIR, 'feat')
        self.assertEqual(Y.shape[0], self.n_cells)
        self.assertEqual(X.shape, (self.n_cells, self.featsize))

        image_1_dict = image_cell_info['image_1']
        self.assertEqual(image_1_dict['start'], 0)
        self.assertEqual(image_1_dict['end'], self.image_1_n_cells)

        image_2_dict = image_cell_info['image_2']
        self.assertEqual(image_2_dict['start'], self.image_1_n_cells)
        self.assertEqual(image_2_dict['end'], self.image_1_n_cells+self.image_2_n_cells)


if __name__ == '__main__':
    unittest.main()

