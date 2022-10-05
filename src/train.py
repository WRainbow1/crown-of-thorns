import os
import re
import argparse

import tensorflow as tf

from utils.utils import (preprocess_data,
                        LabelEncoder,
                        parse_tfrecord_fn)

from utils.model import (get_backbone,
                        RetinaNet,
                        RetinaNetLoss)

def main(model_dir : str,
         data_dir : str,
         train_file : str,
         val_file : str,
         batch_size : int,
         
         ):

    label_encoder = LabelEncoder()

    num_classes = 1
    batch_size = 2

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
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        )
    ]

    autotune = tf.data.AUTOTUNE
    tfrecords_dir = '../data/tfrecords'
    train_dataset = tf.data.TFRecordDataset(f"{data_dir}/{train_file}").map(parse_tfrecord_fn).prefetch(autotune)
    val_dataset = tf.data.TFRecordDataset(f"{data_dir}/{val_file}").map(parse_tfrecord_fn).prefetch(autotune)

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
    train_size = int(re.findall(r'\d+', 'file_00-3443.tfrec'.split('-')[1])[0])
    train_steps_per_epoch = train_size // batch_size

    train_steps = 4 * 100000
    epochs = train_steps // train_steps_per_epoch

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', type=str,
                        help='directory to save the model, should be in a GCP bucket',
                        default = 'retinanet/')
    parser.add_argument('--data_dir', '-dd', type=str,
                        help='training data directory, should be in a GCP bucket',
                        default = 'data/tfrecords')
    parser.add_argument('--train_file', '-tf', type=str,
                        help='training file name',
                        default = 'file_00-3443.tfrec')
    parser.add_argument('--val_file', '-vf', type=str,
                        help='validation file name',
                        default = 'file_01-737.tfrec')
    parser.add_argument('--batch_size', '-bs', type=int,
                        help='size of training batches for model',
                        default = 4)

    args = parser.parse_args()

    main(args.model_dir,
         args.data_dir,
         args.train_file,
         args.val_file,
         args.batch_size)