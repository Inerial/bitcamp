import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler,RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math

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
# print(max(train['SibSp']))  # 8
# print(max(train['Parch']))  # 6

def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort=False, dropna=False) ## nan 값도 안빼서 원그래프에 nan도 표시하게 함
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
        if index in survived.index and index in dead.index :    ## 만약 한쪽에만 있는 index라면 에러가 뜨므로 if문으로 분리
            plt.pie([survived[index], dead[index]], labels=['Survivied', 'Dead'], autopct='%1.1f%%')
        elif index in dead.index :
            plt.pie([dead[index]], labels=[ 'Dead'], autopct='%1.1f%%')
        elif index in survived.index :
            plt.pie([survived[index]], labels=['Survivied'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio')
    plt.show()

# pie_chart('Sex')
## 남자 사망률 81, 여자 26 으로 어마어마한 차이

# pie_chart('Pclass')
#  등급이 높아질수록 생존률이 증가 

test['FamilySize'] = 0
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
train['FamilySize'] = 0
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
## 가족크기 (일단 만들어봄)


# pie_chart('SibSp')
## 형제가 한명도 없을때 사망확률 65퍼
## 형제가 한명일때 46퍼 두명일때 53퍼
## 이후 사망확률이 점점 늘어나 4명일때 83퍼,  5,8명은 100퍼가 떠 오히려 생존확률이 줄어듬을 알수 있었다.
## 하지만 2명 이상부터 데이터 자체의 개수가 너무 적어 정확한 판단이 어려울 수 있다고 느꼈다.
## 2명 이상인 데이터를 묶어 다시 그려보기

train['SibSp_sum'] = train['SibSp'].replace([2,3,4,5,8], [2,2,2,2,2])
test['SibSp_sum'] = test['SibSp'].replace([2,3,4,5,8], [2,2,2,2,2])

# pie_chart('SibSp_sum')
## 0명, 2명이상일때는 사망확률이 높고 1명일때는 낮았다.
## 합쳐줄까 해서 합쳐보았지만 별 차이가 없었다.


# pie_chart('Parch')

train['Parch_sum'] = train['Parch'].replace([2,3,4,5,6], [2,2,0,0,0])
test['Parch_sum'] = test['Parch'].replace([2,3,4,5,6], [2,2,0,0,0])
# pie_chart('Parch_sum')
## 2명 초과는 인원수가 적어 큰 의미 x로 보임 -> 합쳐도 2명의 비율은 크게 차이 안남
## 그래도 사망확률이 매우 높은 4,5,6은 사망이 높은  0명에 합치고 3은 2에 붙여보았다.
## 역시 별 차이는 없었음


## ticket fare cabin embarked
train['Embarked'] = train['Embarked'].fillna('S') # S가 굉장히 많아 nan을 S로 치환
test['Embarked'] = test['Embarked'].fillna('S')
train['Embarked_int'] = train['Embarked'].replace(['S', 'Q', 'C'], [0, 1, 2])
test['Embarked_int'] = test['Embarked'].replace(['S', 'Q', 'C'], [0, 1, 2])

# pie_chart('Embarked_int')

## C만 44퍼 나머지는 60퍼



## 나이 결측값 채워넣기
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())

train['AgeCategory'] = 0
train.loc[train["Age"] > 4, "AgeCategory"] = 1
train.loc[train["Age"] > 14, "AgeCategory"] = 2
train.loc[train["Age"] > 24, "AgeCategory"] = 3
train.loc[train["Age"] > 34, "AgeCategory"] = 4
train.loc[train["Age"] > 44, "AgeCategory"] = 5
train.loc[train["Age"] > 54, "AgeCategory"] = 6
train.loc[train["Age"] > 64, "AgeCategory"] = 7

test['AgeCategory'] = 0
test.loc[test["Age"] > 4, "AgeCategory"] = 1
test.loc[test["Age"] > 14, "AgeCategory"] = 2
test.loc[test["Age"] > 24, "AgeCategory"] = 3
test.loc[test["Age"] > 34, "AgeCategory"] = 4
test.loc[test["Age"] > 44, "AgeCategory"] = 5
test.loc[test["Age"] > 54, "AgeCategory"] = 6
test.loc[test["Age"] > 64, "AgeCategory"] = 7

## 영유아의 경우 생존률이 높음 (우선적으로 구한것으로 보이는 흔적)
## 65세 이상의 경우 90퍼나 되는 사망률을 보임

# pie_chart('AgeCategory')


train['KnownCabin'] = 0
train.loc[train['Cabin'].isnull() == False, "KnownCabin"] = 1

test['KnownCabin'] = 0
test.loc[test['Cabin'].isnull() == False, "KnownCabin"] = 1
## 객실의 경우 결측치가 매우많다.
## 결측치가 아닌 경우, 살아서 어느 객실에 있었는지 증언했을 가능성이 높다.
## 차트 확인시 결측치가 아닌경우 높은 생존율을 보임

pie_chart('KnownCabin')



## 만든 데이터들중 사용할 것들을 x, x_pred로 만들어줌
x = np.array([(train['Sex'] == 'female').astype('float'), ## string변수인 female, male로 저장되있으며로 female => 1 , male => 0으로 변경
               train['Pclass'].astype('float'),
               train['Parch'].astype('float'),
               train['SibSp'].astype('float'),
               train['FamilySize'].astype('float'),
               train['Embarked_int'].astype('float'), 
               train['AgeCategory'].astype('float'),
               train['KnownCabin'].astype('float')]).T
x_pred = np.array([(test['Sex'] == 'female').astype('float'),
                    test['Pclass'].astype('float'), 
                    test['Parch'].astype('float'),
                    test['SibSp'].astype('float'),
                    test['FamilySize'].astype('float'),
                    test['Embarked_int'].astype('float'),
                    test['AgeCategory'].astype('float'),
                    test['KnownCabin'].astype('float')]).T



y = train_survived.values # 생존자 데이터

print(x)
print(y)
print(x_pred)


## pca를 시도한 흔적 , 별로 효과가 없어서 제거
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# x_pred = scaler.transform(x_pred)
# from sklearn.decomposition import PCA
# pca = PCA(5)
# x = pca.fit_transform(x)
# x_pred = pca.transform(x_pred)


## 데이터의 수가 충분하지 않다고 느껴 실제 적합에는 사용치 않는 test데이터를 안만듬
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(
    x, y, random_state=66, train_size = 0.8
)


## 효과가 있어보이는 표준화 적용
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
# x_test = scaler.fit_transform(x_test)
x_pred = scaler.transform(x_pred)


''' ##모델 제작
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(128, input_dim = x.shape[1], activation = 'elu'))
model.add(Dense(128, activation = 'elu'))
model.add(Dense(128, activation = 'elu'))
model.add(Dense(128, activation = 'elu'))
model.add(Dense(128, activation = 'elu'))
model.add(Dense(128, activation = 'elu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])
    

from keras.callbacks import EarlyStopping, ModelCheckpoint
import os, shutil
early = EarlyStopping(monitor='val_loss', patience=30)
if not os.path.isdir('./kaggle/check') : 
    os.mkdir('./kaggle/check')
check = ModelCheckpoint(filepath = './kaggle/check/{epoch:02d}-{val_loss:.4f}.hdf5',
                        save_best_only=True, save_weights_only=False)
## fit
model.fit(x, y, batch_size = 100, epochs = 1000, validation_split = 0.3, callbacks=[early, check])


## best값 폴더 밖으로 빼고 나머지 다 지우는 코드
tmp = os.path.dirname(os.path.realpath(__file__))
bestfile = os.listdir('./kaggle/check')[-1]
shutil.move('./kaggle/check/'+ bestfile, './kaggle/'+ bestfile)
if os.path.isdir(tmp+'\\check') :
    shutil.rmtree(tmp +'\\check')



# best모델 불러오기, model_checkpoint로 저장한 최고의 값(가장 마지막 파일)을 불러와 그 모델로 predict
model = load_model('./kaggle/' + bestfile)

## test데이터를 빼서 안함
# loss, acc = model.evaluate(x_test, y_test)

# print('loss :' ,loss)
# print('acc :' ,acc)

y_pred = model.predict(x_pred)

result = [int(np.round(i)) for i in y_pred.T[0]]  ## 캐글에서 서밋받을때 int형으로 저장해주어야 값을 제데로 받음

# print(result)

##제출용
submission = pd.DataFrame({ "PassengerId": test.index, "Survived": result}) 
# print(submission)

submission.to_csv('./kaggle/csv/submission_rf.csv', index=False) ## 제출용 csv만들기 '''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore')
model = all_estimators(type_filter = 'classifier')

for (name, algorithm) in model:
    model = algorithm()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 ", accuracy_score(y_test, y_pred))