import pandas as pd, numpy as np

data = pd.read_csv('./data/dacon/comp4/201901-202003.csv',)

kind_of_clss = set()
kind_of_clss.update(data['STD_CLSS_NM'])

for clss in list(kind_of_clss):
    tmp_x = data[data['STD_CLSS_NM'] == clss]
    tmp_x = tmp_x[(data['REG_YYMM'] == 201901) | (data['REG_YYMM'] == 201902)| (data['REG_YYMM'] == 201903)]
    tmp_y = data[data['STD_CLSS_NM'] == clss]
    tmp_y = tmp_y[(data['REG_YYMM'] == 202001) | (data['REG_YYMM'] == 202002)| (data['REG_YYMM'] == 202003)]
    print(tmp_x)
    print(tmp_y)


# REG_YYMM  CARD_SIDO_NM    CARD_CCG_NM         STD_CLSS_NM  HOM_SIDO_NM  HOM_CCG_NM   AGE      SEX_CTGO_CD  FLC        CSTMR_CNT      AMT      CNT
# 년월     카드이용지역_시도    카드이용 시군구     업종명        고객 거주지역시도, 시군구 고객나이    성별        가구특징    이용고객수     이용액    이용건수


{'기타 대형 종합 소매업', '중식 음식점업', '슈퍼마켓', '마사지업', '전시 및 행사 대행업', '택시 운송업', '기타 수상오락 서비스업', '그외 기타 스포츠시설 운영업',
 '비알콜 음료점업', '여관업', '기타 외국식 음식점업', '기타 주점업', '여행사업', '욕탕업', '스포츠 및 레크레이션 용품 임대업', '그외 기타 종합 소매업',
  '체인화 편의점', '차량용 주유소 운영업', '일식 음식점업', '서양식 음식점업', '골프장 운영업', '과실 및 채소 소매업', '기타음식료품 위주종합소매업',
   '호텔업', '면세점', '건강보조식품 소매업', '자동차 임대업', '일반유흥 주점업', '한식 음식점업', '빵 및 과자류 소매업', '화장품 및 방향제 소매업',
    '피자 햄버거 샌드위치 및 유사 음식점업', '정기 항공 운송업', '차량용 가스 충전업', '휴양콘도 운영업', '수산물 소매업', '내항 여객 운송업',
     '버스 운송업', '육류 소매업', '그외 기타 분류안된 오락관련 서비스업', '관광 민예품 및 선물용품 소매업'}