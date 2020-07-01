from sklearn.metrics import mean_squared_log_error
import numpy as np

## 제주에 3배 가중치 줘야함 아직 추가 안함
def my_rmsle(y_pred, y_true):
    return "RMSLE", np.sqrt(mean_squared_log_error(y_true, y_pred)), False