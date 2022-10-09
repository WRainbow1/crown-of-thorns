from src import preprocessing, train
import argparse
import tensorflow as tf

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', '-id', type=str,
                        help='directory to save the model, should be in a GCP bucket')
    parser.add_argument('--label_dir', '-ld', type=str,
                        help='training data directory, should be in a GCP bucket')
    parser.add_argument('--label_file', '-lf', type=str,
                        help='training file name')
    parser.add_argument('--tfrecords_dir', '-tfd', type=str,
                        help='validation file name')
    parser.add_argument('--model_dir', '-md', type=str,
                        help='directory to save the model, should be in a GCP bucket')
    parser.add_argument('--train_file', '-tf', type=str,
                        help='training file name')
    parser.add_argument('--val_file', '-vf', type=str,
                        help='validation file name')
    parser.add_argument('--batch_size', '-bs', type=int,
                        help='size of training batches for model')
    parser.add_argument('--epochs', '-e', type=int,
                        help='size of training batches for model')

    args = parser.parse_args()

    preprocessing.main(args.image_dir,
                              args.label_dir,
                              args.label_file,
                              args.tfrecords_dir)

    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

    if len(tf.config.list_physical_devices('GPU')) > 0 and 1==2:

        with tf.device(tf.config.list_physical_devices('GPU')[0].name):

            train.main(args.model_dir,
                           args.tfrecords_dir,
                           args.train_file,
                           args.val_file,
                           args.batch_size,
                           args.epochs)

    else:
        train.main(args.model_dir,
                       args.tfrecords_dir,
                       args.train_file,
                       args.val_file,
                       args.batch_size,
                       args.epochs)