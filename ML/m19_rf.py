from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer()

x_train,x_test,y_train,y_test = train_test_split(
    cancer.data, cancer.target, random_state=66, test_size = 0.2
)
model = RandomForestClassifier(n_jobs=-1, n_estimators=200)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print('acc :', acc)

print(model.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importacnes_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("feature_importace")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importacnes_cancer(model)
plt.show()
