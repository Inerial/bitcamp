from sklearn.manifold import TSNE
from keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

def viz_img(y_pred):
    n = 10
    fig = plt.figure(1)
    box_index = 1
    for cluster in range(10):
        result = np.where(y_pred == cluster)
        for i in np.random.choice(result[0].tolist(), n, replace=False):
            ax = fig.add_subplot(n, n, box_index)
            plt.imshow(x_train[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            box_index += 1
    plt.show()


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)


# model = KMeans(init="k-means++", n_clusters=10, random_state=0)
# model.fit(x_train)
# y_pred = model.labels_
    
# viz_img(y_pred)


# model = TSNE(learning_rate=300)
# transformed = model.fit_transform(x_train)

from sklearn.cluster import DBSCAN
model = DBSCAN(eps=2.4, min_samples=100, n_jobs=6)
predict = model.fit(x_train)
y_pred = predict.labels_

# Assign result to df
# dataset = pd.DataFrame({'Column1':transformed[:,0],'Column2':transformed[:,1]})
# dataset['cluster_num'] = pd.Series(predict.labels_)

viz_img(y_pred)