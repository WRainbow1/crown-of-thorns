import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import argparse

import numpy as np
import tensorflow as tf

import ast

from utils.utils import create_example
from utils.utils import parse_tfrecord_fn

#shape is 720 x 1280

def parse_coords(coord_list):
    if coord_list != []:
        coords =  np.array([list(coord_dict.values()) for coord_dict in coord_list])
        return tf.convert_to_tensor(np.array([(coords[:,0]+coords[:,2]/2)/1280, (coords[:,1]+coords[:,3]/2)/720, coords[:,2]/1280, coords[:,3]/720]).T, dtype=tf.float32)
    else:
        # return tf.convert_to_tensor(np.array([[np.nan, np.nan, np.nan, np.nan]]), dtype=tf.float32)
        return '-'

def parse_labels(array): 
    return np.zeros(array.shape[0], dtype=int)

def main(image_dir : str,
         label_dir : str,
         label_file : str,
         tfrecords_dir : str):

    df = pd.read_csv(f'{label_dir}/{label_file}')

    df['annotations'] = df['annotations'].apply(lambda x: parse_coords(ast.literal_eval(x)))
    df = df[df['annotations']!='-']
    df['label'] = df['annotations'].apply(lambda x: parse_labels(x))
    df['path'] = df['image_id']+'.jpg'

    annots = df.to_dict('records')

    num_samples = len(df)

    train_val_test_split = [0,0.7,0.15,0.15]
    assert sum(train_val_test_split) == 1
    lens = [int(i*num_samples) for i in train_val_test_split]
    indices = [sum(lens[:i+1]) for i, _ in enumerate(lens)]
    indices = [val if val!=indices[i-1] else val+1 for i, val in enumerate(indices)]

    for i, index in enumerate(indices[:-1]):
        samples = annots[index : indices[i+1]]

        with tf.io.TFRecordWriter(
            tfrecords_dir + "/file_%.2i-%i.tfrec" % (i, len(samples))
        ) as writer:
            for sample in samples:
                image_path = f"{image_dir}/{sample['image_id']}.jpg"
                image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                example = create_example(image, image_path, sample)
                writer.write(example.SerializeToString())

    # raw_dataset = tf.data.TFRecordDataset(f"{tfrecords_dir}/{len(samples)}.tfrec")
    # parsed_dataset = raw_dataset.map(parse_tfrecord_fn).prefetch(20)

    # for i, features in enumerate(parsed_dataset.take(1)):
    #     for key in features.keys():
    #         if key != "image" and key != 'objects':
    #             print(f"{key}: {features[key]}")
    #         elif key == 'objects':
    #             a = key

    #     print(f"Image shape: {features['image'].shape}")
    #     plt.figure(figsize=(12, 12))
    #     plt.imshow(features["image"].numpy())
    #     plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', '-id', type=str,
                        help='directory to save the model, should be in a GCP bucket',
                        default = 'data/train_images')
    parser.add_argument('--label_dir', '-ld', type=str,
                        help='training data directory, should be in a GCP bucket',
                        default = 'data')
    parser.add_argument('--label_file', '-lf', type=str,
                        help='training file name',
                        default = 'train.csv')
    parser.add_argument('--tfrecords_dir', '-od', type=str,
                        help='validation file name',
                        default = 'data/tfrecords')

    args = parser.parse_args()

    main(args.image_dir,
         args.label_dir,
         args.label_file,
         args.tfrecords_dir)