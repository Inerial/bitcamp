def outliers(data, axis= 0):
    import numpy as np
    import pandas as pd
    if type(data) == pd.DataFrame:
        data = data.values
    if len(data.shape) == 1:
        quartile_1, quartile_3 = np.percentile(data,[25,75])
        print("1사분위 : ", quartile_1)
        print("3사분위 : ", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)  ## 아래
        upper_bound = quartile_3 + (iqr * 1.5)  ## 위
        return np.where((data > upper_bound) | (data < lower_bound))
    else:
        output = []
        for i in range(data.shape[axis]):
            if axis == 0:
                quartile_1, quartile_3 = np.percentile(data[i, :],[25,75])
            else:
                quartile_1, quartile_3 = np.percentile(data[:, i],[25,75])
            print("1사분위 : ", quartile_1)
            print("3사분위 : ", quartile_3)
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr * 1.5)  ## 아래
            upper_bound = quartile_3 + (iqr * 1.5)  ## 위
            if axis == 0:
                output.append(np.where((data[i, :] > upper_bound) | (data[i, :] < lower_bound))[0])
            else:
                output.append(np.where((data[:, i] > upper_bound) | (data[:, i] < lower_bound))[0])
    return np.array(output)


def split_x(seq, size):
    import numpy as np
    if type(seq) != np.ndarray:
        print("입력값이 array가 아님!")
        return
    elif len(seq.shape) == 1:
        aaa = []
        for i in range(len(seq) - size + 1):
            subset = seq[i:i+size]
            aaa.append(subset)
        print(type(aaa))
        aaa = np.array(aaa)
        return aaa.reshape(aaa.shape[0], aaa.shape[1], 1)
    elif len(seq.shape) == 2:
        aaa = []
        for i in range(len(seq.T) - size + 1):
            subset = seq.T[i:i+size]
            aaa.append(subset)
        print(type(aaa))
        return np.array(aaa)
    else :
        print("입력값이 3차원 이상!")
        return