from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import gzip

import numpy as np
import h5py


class Dataset(object):
    """
    The Dataset class provides a Python class to represent the output of the prerpocessing stage.
    Any algorithm that uses the dataset class interface should be able to take advantage of any
    dataset for which a attalos.dataset.dataset_prep class has been written.
    """

    TEXT_FEAT_TYPES_AVAILABLE = set(['tags', 'captions'])

    def __init__(self, img_feature_filename,
                 text_feature_filename,
                 text_feat_type="tags",
                 tag_transfomer=None,
                 seed=1024):
        self.rng = np.random.RandomState(seed)
        self.img_feature_filename = img_feature_filename
        self.text_feature_filename = text_feature_filename
        self.text_feat_type = text_feat_type
        if self.text_feat_type not in self.TEXT_FEAT_TYPES_AVAILABLE:
            raise NotImplementedError('Feature type {} not yet available. Please choose from: {}'
                                      .format(text_feat_type, ','.join(self.TEXT_FEAT_TYPES_AVAILABLE)))

        self.tag_transformer = tag_transfomer
        self.__load_image_features()
        self.__load_text_features()

    def __load_image_features(self):
        self.img_feature_file = h5py.File(self.img_feature_filename)
        self.image_feats = self.img_feature_file['feats']
        self.image_ids = self.img_feature_file['ids']
        self.img_feat_size = len(self.image_feats[0,:]) # get length of the first feature vector
        self._num_images = len(self.image_ids)

    def __load_text_features(self):
        if self.text_feature_filename.endswith('.gz'):
            input_file = gzip.open(self.text_feature_filename, 'rt')
        else:
            input_file = open(self.text_feature_filename, 'rt')
        self.text_feats = json.load(input_file)[self.text_feat_type]

    def get_index(self, item_index):
        # Transform image index to image_id
        item_id = self.image_ids[item_index]
        # Make sure our id's are strings
        #if not isinstance(item_id, str):
        #    item_id = str(item_id)
        
        try:
            item_id = str(item_id, 'utf-8')
        except:
            raise

        img_feat = self.image_feats[item_index, :]
        # If we are supposed to transform tags using TextTransformer then transform tags
        if self.text_feat_type == 'tags' and self.tag_transformer is not None:
            text_feat = []
            for feat in self.text_feats[item_id]:
                feat = str(feat)
                text_feat.append(self.tag_transformer[feat])
        else:
            text_feat = self.text_feats[item_id]

        return img_feat, text_feat

    def get_next_batch(self, batch_size):
        """
        Get a batch of image and text features (text features will be optionally encoded using the tagtransformer
        Args:
            batch_size: # Of items to get

        Returns:
            image_feats: Image features
            text_feats: List of text features (optionally transformed using tag_transformer)
        """
        # Get random batch of item ids to extract
        items_in_batch = np.random.randint(0, self._num_images, batch_size)

        # Initialize image/text return objects
        img_feats = np.zeros((batch_size, self.img_feat_size))
        text_feats = []

        # For each item to extract from the batch
        for i, item_index in enumerate(items_in_batch):
            img_feat, text_feat = self.get_index(item_index)

            # Add image features to output
            img_feats[i, :] = img_feat
            text_feats.append(text_feat)
        return img_feats, text_feats

    @property
    def num_images(self):
        return self._num_images

    def __getitem__(self, item):
        return self.get_index(item)

    def __iter__(self):
        return iter(range(self._num_images))


def main():
    import argparse
    import time
    import onehot
    parser = argparse.ArgumentParser(description='Exercise Dataset class.')
    parser.add_argument('--image_feature_file',
                      dest='image_feature_file',
                      type=str,
                      help='Image Feature file')
    parser.add_argument('--text_feature_file',
                      dest='text_feature_file',
                      type=str,
                      help='Text Feature file')
    parser.add_argument('--tag_transformer_file',
                      dest='tag_transformer_file',
                      type=str,
                      help='Tag transformer dictionary file')
    args = parser.parse_args()

    # Load OneHot Enocder
    if args.tag_transformer_file:
        # Assumes this is already created using TextTransformer.save_data_mapping
        tag_transformer = onehot.OneHot(args.tag_transformer_file)
    else:
        tag_transformer = None
    t = Dataset(args.image_feature_file, args.text_feature_file, tag_transfomer=tag_transformer)

    start_time = time.time()
    for i in range(10):
        img_feats, text_feats = t.get_next_batch(128)
    print('Time to extract 10 batches: {}'.format(time.time() - start_time))
    
if __name__ == '__main__':
    main()
