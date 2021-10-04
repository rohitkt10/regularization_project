import numpy as np, os, h5py

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
