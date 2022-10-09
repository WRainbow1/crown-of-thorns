import os
import re
import argparse
from google.cloud import storage

import tensorflow as tf
tf.config.run_functions_eagerly(True)

from src.utils.utils import (preprocess_data,
                        LabelEncoder,
                        parse_tfrecord_fn)

from src.utils.model import (get_backbone,
                        RetinaNet,
                        RetinaNetLoss)

import src.utils.config as config

def main(model_dir : str,
         data_dir : str,
         train_file : str,
         val_file : str,
         batch_size : int,
         epochs : int):

    print('starting training')

    local_trainfile = 'trainfile.tfrec'
    local_valfile = 'valfile.tfrec'

    label_encoder = LabelEncoder()

    num_classes = 1

    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [125, 250, 500, 240000, 360000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, resnet50_backbone)

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer)

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join('gs://crown-of-thorns-data', 'model', "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        )
    ]

    client = storage.Client.from_service_account_json('src/utils/creds.json', project = 'crown-of-thorns')
    bucket = client.get_bucket('crown-of-thorns-data')

    blob = bucket.blob(f"{data_dir}/{train_file}")
    blob.download_to_filename(local_trainfile)

    blob = bucket.blob(f"{data_dir}/{train_file}")
    blob.download_to_filename(local_valfile)

    autotune = tf.data.AUTOTUNE
    train_dataset = tf.data.TFRecordDataset(local_trainfile).map(parse_tfrecord_fn)
    val_dataset = tf.data.TFRecordDataset(local_valfile).map(parse_tfrecord_fn)

    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
    train_dataset = train_dataset.shuffle(8 * batch_size)
    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True, padded_shapes=([1024,1280,3], [None, None], [None])
    )
    train_dataset = train_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    )

    train_dataset = train_dataset.prefetch(autotune)
    val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
    val_dataset = val_dataset.padded_batch(
        batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True, padded_shapes=([1024,1280,3], [None, None], [None])
    )
    val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    val_dataset = val_dataset.prefetch(autotune)

    # Uncomment the following lines, when training on full dataset

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )

    print('done')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', type=str,
                        help='directory to save the model, should be in a GCP bucket')
    parser.add_argument('--data_dir', '-dd', type=str,
                        help='training data directory, should be in a GCP bucket')
    parser.add_argument('--train_file', '-tf', type=str,
                        help='training file name')
    parser.add_argument('--val_file', '-vf', type=str,
                        help='validation file name')
    parser.add_argument('--batch_size', '-bs', type=int,
                        help='size of training batches for model')

    args = parser.parse_args()

    main(args.model_dir,
         args.data_dir,
         args.train_file,
         args.val_file,
         args.batch_size)