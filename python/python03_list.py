# 자료형
#1. 리스트

a = [1,2,3,4,5]
b = [1,2,3,'a','b']
print(a) # [1,2,3,4,5]
print(b) # [1,2,3,'a','b']

print(a[0] + a[3]) # 5
print(str(b[0]) + b[3]) # 1a
print(type(a)) # <class 'list'>
print(a[-2]) # 4
print(a[1:3]) # [2,3]

a = [1 , 2 , 3 , ['a','b','c']]
print(a[1]) # 2
print(a[-1]) # ['a','b','c']
print(a[-1][1]) # b

#1-2. 리스트 슬라이싱
a = [1,2,3,4,5]
print(a[:2]) # [1,2]

#1-3. 리스트 슬라이싱
a = [1,2,3]
b = [4,5,6]
print(a + b) # [1,2,3,4,5,6]
## 만약 사람이 계산하듯 [5,7,9]를 만들고 싶다면 numpy.array를 사용한다.

c = [7,8,9,10]
print(a+c) # [1,2,3,7,8,9,10]

print(a * 3) # [1,2,3,1,2,3,1,2,3]

# print(a[2] + 'hi') #에러
print(str(a[2]) + 'hi') # 3hi

f = '5'
print(a[2] + int(f)) # 8
print(str(a[2]) + f) # 35

# 리스트 관련 함수
a = [1,2,3]
a.append(4)
print(a) # [1,2,3,4]

# a = a.append(5)
# print(a) # None #리턴값이 a가 아닌 none이라서 a값이 날아간다.

a = [1,3,4,2]
a.sort()
print(a) # [1,2,3,4]

a.reverse()
print(a) # [4,3,2,1]

print(a.index(3)) # 1 (입력한 값 3의 index 위치를 출력)
print(a.index(1)) # 3

#앞 숫자에 맞는 인덱스 값에 뒷숫자를 삽입, 해당 인덱스값 뒤에있는 값들은 전부 뒤로 밀려남
a.insert(0,7)
print(a) # [7,4,3,2,1]
a.insert(3,3)
print(a) # [7,4,3,3,2,1]

# remove안 인자를 삭제
a.remove(7)
print(a) # [4,3,3,2,1]
a.remove(3)
print(a) # [4,3,2,1]
