import this

## 파이썬 설치과정 생략

## 들여쓰기 : def, for문 등등으로 구분되어야 하는 단락을 중괄호를 사용하지 않고 들여쓰기만으로 구분가능 => 실수하면 에러뜨니 조심
## 단순히 가독성을 위해 들여쓰기를 할수도있으니 구분 잘할것

#모듈
from numpy import array
## from은 어디서 가져온다는, import는 가져올 클래스를

## 함수

def double(x):
    return x*2
## x의 값을 두배해서 리턴하는 함수

def apply_to_one(f):
    return f(1)
##함수를 매개변수로 받는 함수

y = apply_to_one(lambda x:x+4)
##lambda는 함수를 급조해서 보낸다. 해당식은 x + 4를 리턴하는 함수
## 자주 사용하지는 말자

def my_print(message = "my"):
    print(message)
## 매개변수에 값을 안주면 디폴트값인 my를 출력

## 23p