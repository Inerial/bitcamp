#정수형
a = 1
b = 2
c = a + b
print(c) # 3
d = a*b
print(d) # 2
e = a/b
print(e) # 0.5

#실수형

a = 1.1
b = 2.2
c = a+b
print(c) # 3.3000000000000003

d = a*b
print(d) # 2.4200000000000004
e = a/b
print(e) # 0.5

# 문자열
a = "hel"
b = "lo"
c = a + b
print(c) # hello

a = 123
b = "45"
# c = a + b # 에러
# print(c)

a = 123
a = str(a)
print(a) # 123
b = '45'
c = a + b
print(c) # 12345

a = 123
b = '45'
c = a + int(b)
print(c) # 168

# 문자열 연산하기
a = 'abcdefgh'
print(a[0]) # a
print(a[3]) # d
print(a[5]) # f
print(a[-1]) # h
print(a[-2]) # g
print(type(a)) # <class 'str'>

b = 'xyz'
print(a + b) # abcdefghxyz

# 문자열 인덱싱
a = 'Hello, Deep learning'
print(a[7]) # D 
print(a[-1]) # g
print(a[-2]) # n
print(a[3:9]) # lo, De
print(a[3:-5]) # lo, Deep lea

print(a[:-1]) # Hello, Deep learnin

print(a[1:]) # ello, Deep learning
print(a[5:-4]) # , Deep lear

## 음수는 뒤에서부터 세는걸로 해준다