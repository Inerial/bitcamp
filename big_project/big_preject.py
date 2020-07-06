import tensorflow as tf
import numpy as np
import IPython.display as display
import os

data = tf.data.TFRecordDataset('./0A.tfrecord', compression_type="GZIP").map(from_tfrecord)


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.BytesList(value=value))

def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _validate_text(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, 'unicode'):
        return text.encode('utf8', 'ignore')
    else:
        return str(text)
tf.compat.v1.Fix
string_set = tf.train.Example(features=tf.train.Features(feature={
    'video_id': _int64_feature(image.shape[0]),
    'start_time_seconds': _int64_feature(image.shape[1]),
    'end_time_seconds': _bytes_feature(_binary_image),
    'labels': _bytes_feature(_binary_label),
    '': _float_feature(image.mean().astype(np.float32)),
    'std': _float_feature(image.std().astype(np.float32)),
    'filename': _bytes_feature(str.encode(filename)),
}))

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    return image, label, height, width, depth

with tf.Session() as sess:
    filename_queue = tf.train.string_input_producer(["../data/svhn/svhn_train.tfrecords"])
    image, label, height, width, depth = read_and_decode(filename_queue)
    image = tf.reshape(image, tf.pack([height, width, 3]))
    image.set_shape([32,32,3])
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(1000):
        example, l = sess.run([image, label])
        print (example,l)
    coord.request_stop()
    coord.join(threads)