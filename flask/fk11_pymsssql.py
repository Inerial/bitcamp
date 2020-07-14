import pymssql as ms
import numpy as np
# print("잘 접속됬지?")


with ms.connect(server='127.0.0.1', user = 'bit', password='1234',database='bitdb') as conn:
    # 연결한 sql정보 입력

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM sonar;")
    print(np.array(cursor.fetchall()))
    row = cursor.fetchone() ## 햔줄씩 가져온다.
    # for i in range(150):
    #     print(row)
    #     row = cursor.fetchone()
