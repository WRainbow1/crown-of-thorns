from src import preprocessing, train
import argparse
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', '-p', type=int,
                        help='whether or not to perform preprocessing.')
    parser.add_argument('--train', '-t', type=int,
                        help='whether or not to perform training.')
    parser.add_argument('--cloud', '-c', type=int,
                        help='process on cloud or locally.')
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

    if args.preprocess:
        preprocessing.main(args.cloud,
                           args.image_dir,
                           args.label_dir,
                           args.label_file,
                           args.tfrecords_dir)
    else:
        print('not running preprocessing, assuming tfrecords already available')

    if args.train:
        print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
        if len(tf.config.list_physical_devices('GPU')) > 0 and 1==2:

            with tf.device('GPU:0'):
                print('starting GPU')

                train.main(args.cloud,
                        args.model_dir,
                        args.tfrecords_dir,
                        args.train_file,
                        args.val_file,
                        args.batch_size,
                        args.epochs)
        
        else:
            train.main(args.cloud,
            args.model_dir,
            args.tfrecords_dir,
            args.train_file,
            args.val_file,
            args.batch_size,
            args.epochs)
