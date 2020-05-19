#2.  튜플
# 리스트와 거의 같으나, 삭제, 수정불가

#고정값을 줄때 사용 가능
a = (1,2,3) 
b = 1,2,3
print(type(a))
print(type(b))

# a.remove(2) #수정불가
# print(a)
print(a + b) # (1,2,3,1,2,3) # 새 튜플
print(a * 3) # (1,2,3,1,2,3,1,2,3)
# print(a - 3) #수정불가