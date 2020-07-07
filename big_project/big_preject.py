import tensorflow as tf
import numpy as np
import IPython.display as display
import os
from scipy.io.wavfile import write
from sklearn.decomposition import PCA

audio_record = 'D:/Study/big_project/0A.tfrecord'
vid_ids = []
labels = []
start_time_seconds = [] # in secondes
end_time_seconds = []
feat_audio = []
count = 0
for example in tf.python_io.tf_record_iterator(audio_record):
    tf_example = tf.train.Example.FromString(example)
    #print(tf_example)
    vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
    labels.append(tf_example.features.feature['labels'].int64_list.value)
    start_time_seconds.append(tf_example.features.feature['start_time_seconds'].float_list.value)
    end_time_seconds.append(tf_example.features.feature['end_time_seconds'].float_list.value)

    tf_seq_example = tf.train.SequenceExample.FromString(example)
    n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)

    sess = tf.InteractiveSession()
    rgb_frame = []
    audio_frame = []
    # iterate through frames
    for i in range(n_frames):
        audio_frame.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['audio_embedding'].feature[i].bytes_list.value[0],tf.uint8)
                       ,tf.float32).eval())

    sess.close()
    feat_audio.append([])

    feat_audio[count].append(audio_frame)
    count+=1


# for i in range(0,1):
#     print(vid_ids[i])
#     print(labels[i])
#     print(start_time_seconds[i])
#     print(end_time_seconds[i])
#     print(len(feat_audio[i]))
#     print(feat_audio[i][0])
#     print(feat_audio[i][0][0])

# print(count)
# pca = PCA(44110)
# print(pca.fit_transform(np.array(feat_audio[0][0][0]).reshape(1,128)))
# write('./test.wav', 128, pca.fit_transform(np.array(feat_audio[0][0][0]).reshape(128,1)))

# asdf = np.load('D:/Study/hamsu/vggish/vggish_pca_params.npz')
# print(asdf)
# # ['pca_means', 'pca_eigen_vectors']

params = np.load('D:/Study/hamsu/vggish/vggish_pca_params.npz')
pca_matrix = params['pca_eigen_vectors']
pca_means = params['pca_means'].reshape(-1, 1)
pca_matrix_inv = np.linalg.inv(pca_matrix)

clipped_embeddings = np.array(feat_audio[0][0]) / (255.0 / 4.0) - 2.0
# print(clipped_embeddings)
embedding_batch = (np.dot(pca_matrix_inv, clipped_embeddings.T) + pca_means).T
# embedding_tensor = sess.graph.get_tensor_by_name('vggish/embedding' + ':0')
# [embedding_batch] = sess.run([embedding_tensor],feed_dict={features_tensor: input_batch})
# pca_applied = np.dot(pca_matrix,(embeddings_batch.T - pca_means)).T

# clipped_embeddings = np.clip(pca_applied, -2.0, 2.0)



# quantized_embeddings = quantized_embeddings.astype(np.uint8)

