import pandas as pd
import numpy as np

data = pd.read_csv('./data/dacon/comp4/201901-202003.csv')
data = data.fillna('')

df = data.copy()
df = df[['REG_YYMM','CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM','AGE','SEX_CTGO_CD','FLC','AMT']]
df = df.groupby(['REG_YYMM','CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM','AGE','SEX_CTGO_CD','FLC']).sum().reset_index(drop=False)
df_202001 = df.loc[df['REG_YYMM']==202001]
df_201901 = df.loc[df['REG_YYMM']==201901]

df_202002 = df.loc[df['REG_YYMM']==202002]
df_201902 = df.loc[df['REG_YYMM']==201902]

df_202003 = df.loc[df['REG_YYMM']==202003]
df_201903 = df.loc[df['REG_YYMM']==201903]

df_201903 = df_201903[['CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM','AGE','SEX_CTGO_CD','FLC','AMT']]
df_202003 = df_202003[['CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM','AGE','SEX_CTGO_CD','FLC','AMT']]
tmp03 = df_201903.merge(df_202003, left_on=['CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM','AGE','SEX_CTGO_CD','FLC'], right_on=['CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM','AGE','SEX_CTGO_CD','FLC'], how='left')
tmp03 = tmp03.fillna(0)
tmp03['19/20'] = tmp03['AMT_x']/tmp03['AMT_y']
tmp03['20/19'] = tmp03['AMT_y']/tmp03['AMT_x']
tmp03.to_csv('./dacon/comp4/03차이확인.csv', encoding='utf-8-sig')

df_201902 = df_201902[['CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM','AGE','SEX_CTGO_CD','FLC','AMT']]
df_202002 = df_202002[['CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM','AGE','SEX_CTGO_CD','FLC','AMT']]
tmp02 = df_201902.merge(df_202002, left_on=['CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM','AGE','SEX_CTGO_CD','FLC'], right_on=['CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM','AGE','SEX_CTGO_CD','FLC'], how='left')
tmp02 = tmp02.fillna(0)
tmp02['19/20'] = tmp02['AMT_x']/tmp02['AMT_y']
tmp02['20/19'] = tmp02['AMT_y']/tmp02['AMT_x']
tmp02.to_csv('./dacon/comp4/02차이확인.csv', encoding='utf-8-sig')

df_201901 = df_201901[['CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM','AGE','SEX_CTGO_CD','FLC','AMT']]
df_202001 = df_202001[['CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM','AGE','SEX_CTGO_CD','FLC','AMT']]
tmp01 = df_201901.merge(df_202001, left_on=['CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM','AGE','SEX_CTGO_CD','FLC'], right_on=['CARD_SIDO_NM','STD_CLSS_NM','HOM_SIDO_NM','AGE','SEX_CTGO_CD','FLC'], how='left')
tmp01 = tmp01.fillna(0)
tmp01['19/20'] = tmp01['AMT_x']/tmp01['AMT_y']
tmp01['20/19'] = tmp01['AMT_y']/tmp01['AMT_x']
tmp01.to_csv('./dacon/comp4/01차이확인.csv', encoding='utf-8-sig')

submission = pd.read_csv('./data/dacon/comp4/submission.csv', index_col=0)
submission = submission.loc[submission['REG_YYMM']==202004]
submission = submission[['CARD_SIDO_NM', 'STD_CLSS_NM']]
submission = submission.merge(df, left_on=['CARD_SIDO_NM', 'STD_CLSS_NM'], right_on=['CARD_SIDO_NM', 'STD_CLSS_NM'], how='left')
submission = submission.fillna(0)

tmp.to_csv('./dacon/comp4/차이확인.csv', encoding='utf-8-sig')
AMT = list(submission['AMT'])*2

submission = pd.read_csv('./data/dacon/comp4/submission.csv', index_col=0)
submission['AMT'] = AMT
submission.to_csv('./dacon/comp4/comp4_sub.csv', encoding='utf-8-sig')
submission.head()