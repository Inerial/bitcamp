import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

pca = PCA()
X2 = pca.fit_transform(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) ## 각 pca값이 가지는퍼센트

aaa = np.argmax(cumsum >= 0.94) + 1
print(cumsum)
print(aaa)