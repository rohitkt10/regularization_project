"""
Function to download invivo data and save in tfrecords format. 
"""

import numpy as np, os, h5py 
import tensorflow as tf
import subprocess
from tqdm import tqdm

IN_VIVO_NAMES = {
                'GM12878':'https://www.dropbox.com/s/e972vdcwsvhuoca/GM12878.h5', 
                'A549':'https://www.dropbox.com/s/no2x1tfzo1r4pb2/A549.h5', 
                'HeLa-S3':'https://www.dropbox.com/s/01t5aiyvrm0avca/HeLa-S3.h5',
                }

# functions to convert various datatypes into tf.train.Feature objects
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def _get_schema(L, A, num_labels, x, y):
    schema = {
                "length":_int64_feature(L),
                "depth":_int64_feature(A),
                "num_labels":_int64_feature(num_labels),
                "x":_bytes_feature(_serialize_array(x)),
                "y":_bytes_feature(_serialize_array(y))
            }
    return schema

def _encoder(x, y):
    """
    Set up single tf.train.Example instance. 
    """
    L, A = x.shape
    num_labels = y.shape[0]
    schema = _get_schema(L, A, num_labels, x, y)
    features = tf.train.Features(feature=schema)
    example = tf.train.Example(features=features)
    return example

def _write_data_to_tfrecord(x, y, 
                           fname="data",
                           compression=False,
                           compression_level=4):
    """Takes a dataset (or a shard of a dataset) and writes it 
    to a tfrecords file with specified compression settings"""
    if not fname.lower().endswith(".tfrecords"):
        fname= fname+".tfrecords"

    # set up the tfrecords writer
    if compression:
        options = tf.io.TFRecordOptions(compression_type="GZIP", compression_level=compression_level)
    else:
        options = None
    writer = tf.io.TFRecordWriter(fname, options=options)

    # iterate over all samples in the dataset 
    for i in range(len(x)):
        example = _encoder(x[i], y[i]).SerializeToString()  
        writer.write(example)  
    writer.close()

def _h5py_to_tfrecords(
                    fpath,
                    savepath="./data",
                    shard_size=None,
                    compression=True,
                    compression_level=4,
                    splits=['train', 'test', 'valid'],
                    variable_names=['x', 'y'],
                      ):
    """
    Arguments
    ---------
    1. fpath <str> - path to h5 file.
    2. savepath <str> - directory path where tfrecords are to be saved.
    3. shard_size <optional, int> - Number of samples per tfrecord shards.
    4. compression <bool> - Whether to apply compression.
    5. compression_level <int> - Degree of compression if compression is to be applied.
    6. splits <list of str> - Names of splits.
    7. variable_names <list of str> - Names of input and output variables in order (x, y default).

    Returns
    -------
    None
    """
    assert fpath.endswith('.h5') or fpath.endswith('.hdf5')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    assert len(splits) and len(variable_names)
    
    # load file
    f = h5py.File(fpath, 'r')
    keys = list(f.keys())

    # write tfrecords for all splits
    for split in splits:
        split_keys = [_k for _k in keys if split in _k]
        v1key = [_k for _k in split_keys if variable_names[0] in _k][0]
        nsamples = f[v1key].shape[0]

        # set up the shard size
        if shard_size:
            nshards = nsamples // shard_size
            if nsamples % shard_size != 0:
                nshards += 1
        else:
            shard_size = nsamples
            nshards = 1
        
        # loop over all shards 
        c = 0 # this is the current start index 
        for i in tqdm(range(nshards)):
            if nshards > 1:
                shard_name =  f'{split}_{i+1}_{nshards}.tfrecords'
                shard_name = os.path.join(savepath, split, shard_name)
            else:
                shard_name =  f'{split}.tfrecords'
                shard_name = os.path.join(savepath, shard_name)
            shard_dir = os.path.dirname(shard_name)
            if not os.path.exists(shard_dir):
                os.makedirs(shard_dir)
            
            # get the variable keys for current split
            xkey = [_k for _k in split_keys if variable_names[0] in _k][0]
            ykey = [_k for _k in split_keys if variable_names[1] in _k][0]
            if nsamples - c >= shard_size:
                current_shard_size = shard_size
            else:
                current_shard_size = nsamples - c
            xshard = tf.cast(f[xkey][c:c+current_shard_size][:], tf.uint8)
            yshard = tf.cast(f[ykey][c:c+current_shard_size][:], tf.uint8)
            c += current_shard_size

            _write_data_to_tfrecord(
                                xshard, 
                                yshard, 
                                shard_name, 
                                compression=compression, 
                                compression_level=compression_level
                                )
    f.close()

def download(name='A549', savepath="./data", shard_size=None, **kwargs):
    assert name in IN_VIVO_NAMES
    hlink = IN_VIVO_NAMES[name]
    fpath = name+".h5"
    

    # download the data in h5 format
    if not os.path.exists(fpath):
        process = subprocess.Popen(
                            ['wget', hlink], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE
                                )
        stdout, stderr = process.communicate()
    
    
    _h5py_to_tfrecords(
                    fpath=fpath,
                    savepath=savepath,
                    shard_size=shard_size,
                    compression=True,
                    **kwargs,
    )

    # delete the downloaded h5 file
    process = subprocess.Popen(
                        ['rm', '-rf', fpath],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                            )
    return





