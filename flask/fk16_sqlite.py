import sqlite3, os, datetime

videoPath = 'D:/Study/data/team/video'
videos = os.listdir(videoPath)
print(videos)
conn = sqlite3.connect("D:/Study/data/team/prac_my.db")

cursor = conn.cursor()

# cursor.execute("""CREATE TABLE IF NOT EXISTS supermarket(Itemno INTEGER, Category TEXT,
#                 FoodName TEXT, Company TEXT, Price INTEGER)""")

# sql = "DELETE FROM supermarket" 
# cursor.execute(sql)
for i,video in enumerate(videos):
       sql = "INSERT into video(video_path,upload_time) \
              values (?,?)"
       cursor.execute(sql, (videoPath+'/'+video, datetime.datetime.now()))


# sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) \
#        values (?,?,?,?,?)"
# cursor.execute(sql, (2, '음료수', '망고주스', '편의점', 1000))


# sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) \
#        values (?,?,?,?,?)"
# cursor.execute(sql, (3, '고기', '소고기', '하나로마트', 10000))

# sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) \
#        values (?,?,?,?,?)"
# cursor.execute(sql, (4, '박카스', '약', '약국', 500))

# sql = "SELECT * FROM supermarket"
# # sql = "SELECT Itemno, Category, FoodName, Company, Price FROM supermarket"
# cursor.execute(sql)

# rows = cursor.fetchall()

# for row in rows:
#     print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " "+ 
#               str(row[3]) + " " + str(row[4]))

conn.commit()
conn.close()