def sum1(a, b):
    return a + b

a = 10
b = 12

c = sum1(a,b)

print(c)

### 곱셈, 나눗셈, 뺄셈을 만드시오

def mul1(a,b):
    return a*b

def div1(a,b):
    return a/b

def sub1(a,b):
    return a-b

print(mul1(a,b))
print(div1(a,b))
print(sub1(a,b))

def sayYeh():
    return 'Hi'

aaa = sayYeh()
print(aaa)

def sum1(a=0,b=0,c=0):
    return a + b + c

a = 1
b = 2
c = 34
d = sum1(a,b,c)
print(d)
sum1(1,2,3)