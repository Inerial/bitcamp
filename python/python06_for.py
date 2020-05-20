# 4. 반복문

a = 0
'''
for(i = 1 ,i = 100,i++){
    a = a + i
} # 옛날 구닥다리 방식
'''
a = {'name': 'yun', 'phone' : '010', 'birth' : '0511'}
for i in a.keys():
    print(i == a[i])
#name
#phone
#birth

a = [1,2,3,4,5,6,7,8,9,10]
#a 리스트의 인자의 개수만큼 돌리기
for i in a:
    print(i,':' , i*i)

#print('melong')

for i in a:
    print(i)

## while문

'''
while 조건문:
    수행할 문장
'''

### if문
if 1 :
    print('True')
else : print('False')


if 3 :
    print('True')
else : print('False')

if 0 :
    print('True')
else : print('False')

if -1 :
    print('True')
else : print('False')

#비교연산자
## ==, >, < , >=, <=, != < 

a = 1
if a == 1:
    print('출력 잘됨')

money = 10000
if money >= 30000:
    print('한우 먹자')
else:
    print('라면 먹자')

## 조건연산자
# and, or,not ,xor

money = 20000
card = 1
if money >= 30000 or card == 1:
    print('한우먹자')
else :
    print("라면먹자")

jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i >= 60:
        number = number + 1
        print("경]합격[축")
print("합격인원 :", number, "명")

##############################################
# break, continue

jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i < 30:
        break
    if i >= 60:
        number = number + 1
        print("경]합격[축")
print("합격인원 :", number, "명")


print("==========continue============")
jumsu = [90, 25, 67, 45, 80]
number = 0
for i in jumsu:
    if i < 60:
        continue
    if i >= 60:
        number = number + 1
        print("경]합격[축")
print("합격인원 :", number, "명")

