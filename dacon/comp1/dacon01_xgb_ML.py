import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error as mae
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)

x_train = np.load('./dacon/comp1/x_train.npy')
y_train = np.load('./dacon/comp1/y_train.npy')
x_pred = np.load('./dacon/comp1/x_pred.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 66
)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

# 2. model

parameters =[
    {'n_estimators': [3000],
    'learning_rate': [0.1],
    'max_depth': [6], 
    'booster': ['dart'], 
    'rate_drop' : [0.2],
    'eval_metric': ['logloss','mae'], 
    'is_training_metric': [True], 
    'max_leaves': [144], 
    'colsample_bytree': [0.8], 
    'subsample': [0.8],
    'seed': [66]
    }
]
kfold = KFold(n_splits=5, shuffle=True, random_state=66)
y_test_pred = []
y_pred = []
search = RandomizedSearchCV(XGBRegressor(n_jobs=6), parameters, cv = kfold, n_iter=1)

for i in range(4):
    fit_params = {
        'verbose': True,
        'eval_metric': ['logloss','mae'],
        'eval_set' : [(x_train,y_train[:,i]),(x_test,y_test[:,i])],
        'early_stopping_rounds' : 5
    }
    search.fit(x_train, y_train[:,i],**fit_params)
    y_pred.append(search.predict(x_pred))
    y_test_pred.append(search.predict(x_test))
    print(search.best_score_)


y_pred = np.array(y_pred).T
y_test_pred = np.array(y_test_pred).T

print(y_pred.shape)
r2 = r2_score(y_test,y_test_pred)
mae = mae(y_test,y_test_pred)
print('r2 :', r2)
print('mae :', mae)

# submissions = pd.DataFrame({
#     "id": test.index,
#     "hhb": y_pred[0,:],
#     "hbo2": y_pred[1,:],
#     "ca": y_pred[2,:],
#     "na": y_pred[3,:]
# })

# submissions.to_csv('./dacon/comp1/comp1_sub.csv', index = False)





# [248]   validation_0-mae:3.53508        validation_1-mae:3.51367
# [249]   validation_0-mae:3.53534        validation_1-mae:3.51377
# [250]   validation_0-mae:3.53574        validation_1-mae:3.51393
# [251]   validation_0-mae:3.53613        validation_1-mae:3.51418
# [252]   validation_0-mae:3.53624        validation_1-mae:3.51417
# [253]   validation_0-mae:3.53632        validation_1-mae:3.51409
# Stopping. Best iteration:
# [223]   validation_0-mae:3.52965        validation_1-mae:3.51197

# r2 : -0.3008520458571647
# PS D:\Study>  cd 'd:\Study'; ${env:PYTHONIOENCODING}='UTF-8'; ${env:PYTHONUNBUFFERED}='1'; & 'C:\Users\bitcamp\anaconda3\python.exe' 'c:\Users\bitcamp\.vscode\extensions\ms-python.python-2020.6.89148\pythonFiles\ptvsd_launcher.py' '--default' '--nodebug' '--client' '--host' 'localhost' '--port' '58277' 'd:\Study\dacon\comp1\dacon01_xgb_ML.py' 
# (8000, 176)
# (8000, 4)
# (2000, 176)
# C:\Users\bitcamp\anaconda3\lib\site-packages\xgboost\core.py:444: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
#   "because it will generate extra copies and increase " +
# C:\Users\bitcamp\anaconda3\lib\site-packages\xgboost\core.py:444: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
#   "because it will generate extra copies and increase " +
# C:\Users\bitcamp\anaconda3\lib\site-packages\xgboost\core.py:444: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
#   "because it will generate extra copies and increase " +
# C:\Users\bitcamp\anaconda3\lib\site-packages\xgboost\core.py:444: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
#   "because it will generate extra copies and increase " +
# r2 : 0.5212904588126568


# [239]   validation_0-logloss:-179.28279 validation_0-rmse:4.63913       validation_1-logloss:-182.43228 validation_1-rmse:4.62343
# [240]   validation_0-logloss:-179.23657 validation_0-rmse:4.63940       validation_1-logloss:-182.43274 validation_1-rmse:4.62325
# [241]   validation_0-logloss:-179.23595 validation_0-rmse:4.63964       validation_1-logloss:-182.43219 validation_1-rmse:4.62323
# [242]   validation_0-logloss:-179.23982 validation_0-rmse:4.63970       validation_1-logloss:-182.43188 validation_1-rmse:4.62327
# [243]   validation_0-logloss:-179.20995 validation_0-rmse:4.63982       validation_1-logloss:-182.43199 validation_1-rmse:4.62330
# [244]   validation_0-logloss:-179.20387 validation_0-rmse:4.64020       validation_1-logloss:-182.42915 validation_1-rmse:4.62366
# [245]   validation_0-logloss:-179.21519 validation_0-rmse:4.64030       validation_1-logloss:-182.42902 validation_1-rmse:4.62347
# [246]   validation_0-logloss:-179.21292 validation_0-rmse:4.64052       validation_1-logloss:-182.42825 validation_1-rmse:4.62363
# [247]   validation_0-logloss:-179.15292 validation_0-rmse:4.64088       validation_1-logloss:-182.42778 validation_1-rmse:4.62388
# [248]   validation_0-logloss:-179.15204 validation_0-rmse:4.64103       validation_1-logloss:-182.42763 validation_1-rmse:4.62378
# [249]   validation_0-logloss:-179.17821 validation_0-rmse:4.64124       validation_1-logloss:-182.42764 validation_1-rmse:4.62382
# [250]   validation_0-logloss:-179.17889 validation_0-rmse:4.64151       validation_1-logloss:-182.42778 validation_1-rmse:4.62394
# [251]   validation_0-logloss:-179.17790 validation_0-rmse:4.64181       validation_1-logloss:-182.42793 validation_1-rmse:4.62414
# [252]   validation_0-logloss:-179.17780 validation_0-rmse:4.64187       validation_1-logloss:-182.42793 validation_1-rmse:4.62411
# [253]   validation_0-logloss:-179.18610 validation_0-rmse:4.64196       validation_1-logloss:-182.42787 validation_1-rmse:4.62410
# [254]   validation_0-logloss:-179.17529 validation_0-rmse:4.64218       validation_1-logloss:-182.42752 validation_1-rmse:4.62401
# [255]   validation_0-logloss:-179.16809 validation_0-rmse:4.64253       validation_1-logloss:-182.42548 validation_1-rmse:4.62430
# [256]   validation_0-logloss:-179.17600 validation_0-rmse:4.64258       validation_1-logloss:-182.42535 validation_1-rmse:4.62436
# Stopping. Best iteration:
# [226]   validation_0-logloss:-179.61789 validation_0-rmse:4.63691       validation_1-logloss:-182.50270 validation_1-rmse:4.62245

# r2 : -0.08807016636663034
# PS D:\Study>


# [980]   validation_0-logloss:-168.99028 validation_0-rmse:4.76155       validation_1-logloss:-179.79201 validation_1-rmse:4.65580
# [981]   validation_0-logloss:-168.98778 validation_0-rmse:4.76171       validation_1-logloss:-179.79192 validation_1-rmse:4.65590
# [982]   validation_0-logloss:-168.98755 validation_0-rmse:4.76186       validation_1-logloss:-179.79195 validation_1-rmse:4.65590
# [983]   validation_0-logloss:-168.96217 validation_0-rmse:4.76198       validation_1-logloss:-179.79147 validation_1-rmse:4.65602
# [984]   validation_0-logloss:-168.95068 validation_0-rmse:4.76222       validation_1-logloss:-179.79150 validation_1-rmse:4.65606
# [985]   validation_0-logloss:-168.91328 validation_0-rmse:4.76238       validation_1-logloss:-179.79216 validation_1-rmse:4.65621
# [986]   validation_0-logloss:-168.81818 validation_0-rmse:4.76262       validation_1-logloss:-179.79079 validation_1-rmse:4.65622
# [987]   validation_0-logloss:-168.81766 validation_0-rmse:4.76270       validation_1-logloss:-179.79074 validation_1-rmse:4.65620
# [988]   validation_0-logloss:-168.81894 validation_0-rmse:4.76280       validation_1-logloss:-179.79068 validation_1-rmse:4.65626
# [989]   validation_0-logloss:-168.84932 validation_0-rmse:4.76293       validation_1-logloss:-179.79050 validation_1-rmse:4.65637
# [990]   validation_0-logloss:-168.84009 validation_0-rmse:4.76298       validation_1-logloss:-179.79034 validation_1-rmse:4.65639
# [991]   validation_0-logloss:-168.83327 validation_0-rmse:4.76306       validation_1-logloss:-179.63841 validation_1-rmse:4.65644
# [992]   validation_0-logloss:-168.84961 validation_0-rmse:4.76326       validation_1-logloss:-179.77420 validation_1-rmse:4.65659
# [993]   validation_0-logloss:-168.82613 validation_0-rmse:4.76343       validation_1-logloss:-179.77290 validation_1-rmse:4.65648
# [994]   validation_0-logloss:-168.82506 validation_0-rmse:4.76354       validation_1-logloss:-179.78940 validation_1-rmse:4.65651
# [995]   validation_0-logloss:-168.64165 validation_0-rmse:4.76382       validation_1-logloss:-179.78775 validation_1-rmse:4.65654
# [996]   validation_0-logloss:-168.64177 validation_0-rmse:4.76389       validation_1-logloss:-179.78772 validation_1-rmse:4.65660
# [997]   validation_0-logloss:-168.67120 validation_0-rmse:4.76397       validation_1-logloss:-179.78844 validation_1-rmse:4.65669
# [998]   validation_0-logloss:-168.68016 validation_0-rmse:4.76414       validation_1-logloss:-179.63202 validation_1-rmse:4.65674
# [999]   validation_0-logloss:-168.68082 validation_0-rmse:4.76424       validation_1-logloss:-179.63272 validation_1-rmse:4.65682
# r2 : 0.5212904588126568
# PS D:\Study>
# PS D:\Study>

# 5]     validation_0-logloss:-183.21127 validation_0-rmse:4.79449
# [6]     validation_0-logloss:-183.48528 validation_0-rmse:4.73877
# [7]     validation_0-logloss:-183.42357 validation_0-rmse:4.70567
# [8]     validation_0-logloss:-182.18829 validation_0-rmse:4.67783
# [9]     validation_0-logloss:-182.11598 validation_0-rmse:4.66300
# [10]    validation_0-logloss:-182.15427 validation_0-rmse:4.65618
# [11]    validation_0-logloss:-182.11752 validation_0-rmse:4.65135
# [12]    validation_0-logloss:-182.19249 validation_0-rmse:4.64891
# [13]    validation_0-logloss:-182.37483 validation_0-rmse:4.64456
# [14]    validation_0-logloss:-182.17171 validation_0-rmse:4.64987
# [15]    validation_0-logloss:-182.17297 validation_0-rmse:4.64347
# [16]    validation_0-logloss:-182.03308 validation_0-rmse:4.64649
# [17]    validation_0-logloss:-181.63542 validation_0-rmse:4.64893
# [18]    validation_0-logloss:-181.55040 validation_0-rmse:4.64985
# [19]    validation_0-logloss:-181.55051 validation_0-rmse:4.64973
# [20]    validation_0-logloss:-181.54842 validation_0-rmse:4.65068
# Stopping. Best iteration:
# [15]    validation_0-logloss:-182.17297 validation_0-rmse:4.64347

# (10000, 4)
# r2 : -0.09990461313148805
# PS D:\Study> 