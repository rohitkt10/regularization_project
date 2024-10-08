import numpy as np, os, h5py, glob
import tensorflow as tf
IN_VIVO_NAMES = ['A549', 'GM12878', 'HeLa-S3']

def _get_synthetic_data(datadir):
    """
    This function expects to find a file named `synthetic_code_dataset.h5`
    in the given data directory.
    """
    filepath = os.path.join(datadir, 'synthetic_code_dataset.h5')
    with h5py.File(filepath, 'r') as dataset:
        x_train = np.array(dataset['X_train']).astype(np.float32)
        y_train = np.array(dataset['Y_train']).astype(np.float32)
        x_valid = np.array(dataset['X_valid']).astype(np.float32)
        y_valid = np.array(dataset['Y_valid']).astype(np.int32)
        x_test = np.array(dataset['X_test']).astype(np.float32)
        y_test = np.array(dataset['Y_test']).astype(np.int32)
        model_test = np.array(dataset['model_test']).astype(np.float32)
    model_test = model_test.transpose([0,2,1])
    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    x_test = x_test.transpose([0,2,1])
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), model_test


def get_invivo_data(datadir, name='A549'):
    filepath = os.path.join(datadir, f'{name}.h5')
    assert os.path.exists(filepath), 'No such file available.'

    with h5py.File(filepath, 'r') as dataset:
        x_train = np.array(dataset['x_train']).astype(np.float32)
        y_train = np.array(dataset['y_train']).astype(np.float32)
        x_valid = np.array(dataset['x_valid']).astype(np.float32)
        y_valid = np.array(dataset['y_valid']).astype(np.float32)
        x_test = np.array(dataset['x_test']).astype(np.float32)
        y_test = np.array(dataset['y_test']).astype(np.float32)
    
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def get_invivo_tfrecords_data(datadir):
    def _decoder(example):
        """
        Decode a single example.
        """
        schema = {
            'length':tf.io.FixedLenFeature([], tf.int64),
            'depth':tf.io.FixedLenFeature([], tf.int64),
            "num_labels":tf.io.FixedLenFeature([], tf.int64),
            'y':tf.io.FixedLenFeature([], tf.string),
            'x':tf.io.FixedLenFeature([], tf.string),
        }
        content = tf.io.parse_single_example(example, schema)
        length, depth, num_labels = content['length'], content['depth'], content['num_labels']
        x = tf.io.parse_tensor(content['x'], out_type=tf.uint8)
        x = tf.reshape(x, shape=(length, depth))
        x = tf.cast(x, tf.float32)
        y = tf.io.parse_tensor(content['y'], out_type=tf.uint8)
        y = tf.reshape(y, shape=(num_labels,))
        y = tf.cast(y, dtype=x.dtype)
        return  x, y
    
    trainfiles = glob.glob(f"{datadir}/**/train*.tfrecords", recursive=True)
    traindata = tf.data.TFRecordDataset(
                            filenames=trainfiles, 
                            compression_type='GZIP')
    traindata = traindata.map(_decoder)

    validfiles = glob.glob(f"{datadir}/**/valid*.tfrecords", recursive=True)
    validdata = tf.data.TFRecordDataset(
                            filenames=validfiles, 
                            compression_type='GZIP')
    validdata = validdata.map(_decoder)

    testfiles = glob.glob(f"{datadir}/**/test*.tfrecords", recursive=True)
    testdata = tf.data.TFRecordDataset(
                            filenames=testfiles, 
                            compression_type='GZIP')
    testdata = testdata.map(_decoder)

    return traindata, validdata, testdata



    
