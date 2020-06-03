import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler,RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

## 데이터 로드
train = pd.read_csv('./kaggle/csv/train.csv', header = 0, index_col=0, sep=',')
test = pd.read_csv('./kaggle/csv/test.csv', header = 0, index_col=0, sep=',')

## survived 값 분리
train_survived = train.iloc[:,0] 
train = train.iloc[:,1:]

# print(train.shape) ## (891, 10)
# print(survived.shape) ##(891,)
# print(test.shape) ## (418, 10)

# print(train.columns) ## Index(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare','Cabin', 'Embarked'],dtype='object')
# print(test.columns) ## Index(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare','Cabin', 'Embarked'],dtype='object')
## Pclass : 객실 등급
## name : 이름
## Sex : 성별
## Age : 나이
## SibSp : 형제/ 배우자의 탑승 수 (본인 제외)
## Parch : 부모 / 자식의 탑승 수 (본인 제외)
## Ticket : 티켓 번호
## Fare : 요금
## Cabin : 객실 번호
## Embarked : 항만
## 둘의 column이 완전히 동일함을 확인
print(max(train['SibSp']))  # 8
print(max(train['Parch']))  # 6

def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[(train_survived == 1).values][feature].value_counts()
    dead = train[(train_survived == 0).values][feature].value_counts()
    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()

    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
        if len(survived.index) >= i+1 and len(dead.index) >= i+1 :
            plt.pie([survived[index], dead[index]], labels=['Survivied', 'Dead'], autopct='%1.1f%%')
        elif len(survived.index) < i+1 and len(dead.index) >= i+1 :
            plt.pie([dead[index]], labels=[ 'Dead'], autopct='%1.1f%%')
        elif len(survived.index) >= i+1 and len(dead.index) < i+1 :
            plt.pie([survived[index]], labels=['Survivied'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio')
    plt.show()

pie_chart('Sex')

pie_chart('Pclass')

pie_chart('SibSp')
## 형제가 한명도 없을때 사망확률 65퍼
## 형제가 한명일때 46퍼 두명일대 53퍼
## 이후 사망확률이 점점 늘어나 4명일때 83퍼,  5,8명은 100퍼가 떠 오히려 생존확률이 줄어듬을 알수 있었다.
## 하지만 2명 이상부터 데이터 자체의 개수가 너무 적어 정확한 판단이 어려울 수 있다고 느꼈다.
## 2명 이상인 데이터를 묶어 다시 그려보기

train['SibSp_sum'] = train['SibSp'].replace([2,3,4,5,8], [2,2,2,2,2])
print(max(train['SibSp_sum']))
pie_chart('SibSp_sum')
## 0명 65퍼, 1명 46퍼, 2명이상 73퍼로 유의미해 보이는 차이