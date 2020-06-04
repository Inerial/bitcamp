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
    feature_ratio = train[feature].value_counts(sort=False, dropna=False)
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
        if index in survived.index and index in dead.index :
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

# pie_chart('SibSp')
## 형제가 한명도 없을때 사망확률 65퍼
## 형제가 한명일때 46퍼 두명일대 53퍼
## 이후 사망확률이 점점 늘어나 4명일때 83퍼,  5,8명은 100퍼가 떠 오히려 생존확률이 줄어듬을 알수 있었다.
## 하지만 2명 이상부터 데이터 자체의 개수가 너무 적어 정확한 판단이 어려울 수 있다고 느꼈다.
## 2명 이상인 데이터를 묶어 다시 그려보기

train['SibSp_sum'] = train['SibSp'].replace([2,3,4,5,8], [2,2,2,2,2])
test['SibSp_sum'] = test['SibSp'].replace([2,3,4,5,8], [2,2,2,2,2])

# pie_chart('SibSp_sum')
## 사망퍼센트가 높은것을 0, 낮은것을 1에 섞었다.

# pie_chart('Parch')

train['Parch_sum'] = train['Parch'].replace([2,3,4,5,6], [2,2,2,2,2])
test['Parch_sum'] = test['Parch'].replace([2,3,4,5,6], [2,2,2,2,2])

# pie_chart('Parch_sum')
## 2명 초과는 인원수가 적어 큰 의미 x로 보임 -> 합쳐도 2명의 비율은 크게 차이 안남

## ticket fare cabin embarked
train['Embarked'] = train['Embarked'].fillna('S') # S가 굉장히 많아 nan을 S로 치환
test['Embarked'] = test['Embarked'].fillna('S')
train['Embarked_int'] = train['Embarked'].replace(['S', 'Q', 'C'], [0, 1, 2])
test['Embarked_int'] = test['Embarked'].replace(['S', 'Q', 'C'], [0, 1, 2])

# pie_chart('Embarked_int')

## C만 44퍼 나머지는 60퍼

train['AgeBand'] = pd.cut(train['Age'], 8)
test['AgeBand'] = pd.cut(test['Age'], 8)
# pie_chart('AgeBand')

print(train[(train['Age'] <= 10).values]['Parch'].values)
print(train[(train['Age'] > 10).values])


''' x = pd.DataFrame([train['Sex'] == 'female',train['Pclass'], train['SibSp_sum'],
                     train['Parch_sum'], train[Embarked_int]]).T.values
x_pred = pd.DataFrame([test['Sex'] == 'female',test['Pclass'],test['SibSp_sum'],
                     test['Parch_sum'], test[Embarked_int]]).T.values

##0이면 남자, 1이면 여자


y = train_survived.values # survived

print(x)
print(y)
print(x_pred)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(
    x, y, random_state=66, train_size = 0.8
)


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)
# x_pred = scaler.transform(x_pred)


##모델 제작
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(200, input_dim = x.shape[1], activation = 'elu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation = 'elu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation = 'elu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation = 'elu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation = 'elu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation = 'elu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation = 'elu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation = 'elu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation = 'elu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])
    

from keras.callbacks import EarlyStopping, ModelCheckpoint
import os, shutil
early = EarlyStopping(monitor='val_loss', patience=20)
if not os.path.isdir('./kaggle/check') : 
    os.mkdir('./kaggle/check')
check = ModelCheckpoint(filepath = './kaggle/check/{epoch:02d}-{val_loss:.4f}.hdf5',
                        save_best_only=True, save_weights_only=False)
## fit
model.fit(x_train, y_train, epochs = 100, validation_split = 0.3, callbacks=[early, check])

## best값 폴더 밖으로 빼고 나머지 다 지우는 코드
tmp = os.path.dirname(os.path.realpath(__file__))
bestfile = os.listdir('./kaggle/check')[-1]
shutil.move('./kaggle/check/'+ bestfile, './kaggle/'+ bestfile)
if os.path.isdir(tmp+'\\check') :
    shutil.rmtree(tmp +'\\check')

# best모델 불러오기
model = load_model('./kaggle/' + bestfile)

loss, acc = model.evaluate(x_test, y_test)

print('loss :' ,loss)
print('acc :' ,acc)

y_pred = model.predict(x_pred)

result = [int(np.round(i)) for i in y_pred.T[0]]

# print(result)


##제출용
submission = pd.DataFrame({ "PassengerId": test.index, "Survived": result}) 
# print(submission)

submission.to_csv('./kaggle/csv/submission_rf.csv', index=False) '''