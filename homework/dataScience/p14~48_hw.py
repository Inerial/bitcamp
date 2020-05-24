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


try:
    print(0/0)
except ZeroDivisionError:
    print("cannot divide by zero")

## 0으로 나누게 될때 에러메세지 출력

integer_list = [1,2,3]
heterogeneous_list = ["string", 0.1, True]
list_of_lists = [integer_list, heterogeneous_list, []]

list_length = len(integer_list)
list_sum = sum(integer_list)

## 각각 배열의 길이와 배열 값들의 합

x = [0,1,2,3,4,5,6,7,8,9]

zero = x[0]
one = x[1]
nine = x[-1]
eight = x[-2]
x[0] = -1

first_three = x[:3]
three_end = x[3:]
one_to_four = x[1:5]
last_three = x[-3:]
without_first_and_last = x[1:-1]
copy_of_x = x[:]

## 배열의 인덱스 주는법

every_third = x[::3]
five_to_three = x[5:2:-1]
## 세번째는 간격설정

1 in [1,2,3]
0 in [1,2,3]

x = [1,2,3]
x.extend([4,5,6])
#리스트에 더해주기

x = [1,2,3]
y = x + [4,5,6]
## x를 건드리지 않고 y를 추가

x = [1,2,3]
x.append(0)
y = x[-1]
z = len(x)
## 맨뒤에 항목 하나 추가

x,y = [1,2]
_,y = [1,2] ## 첫값은 무시하고 두번째 값만  y에 받음


my_list = [1,2]
my_tuple = (1,2)
other_tuple = 3,4
my_list[1] = 3

try:
    my_tuple[1] = 3
except TypeError:
    print("cannot modify a tuple")
## 튜플은 데이터 수정시 에러

def sum_and_product(x,y):
    return (x+y), (x*y)

sp = sum_and_product(2,3)  ## 한번에 sp에 넣음
s, p = sum_and_product(5,10) ## 따로따로  s, p에 순차대로 넣음

x,y = 1,2
x,y = y,x ## 데이터교환

empty_dict = {}
empty_dict2 = dict()
grades = {"Joel" : 80, "Tim" : 95}

joels_grade = grades["Joel"]

try:
    kates_grade = grades["Kate"]
except KeyError:
    print("no grade for Kate")

joel_has_grade = "Joel" in grades
kate_has_grade = "Kate" in grades
## grades 내부에 왼쪽 값이 존재하는지 확인

joels_grade = grades.get("Joel", 0)
kates_grade = grades.get("Kate", 0)
no_ones_grade = grades.get("No One")
## 각 key값이 가진 value를 리턴

grades["Tim"] = 99
grades["Kate"] = 100
num_students = len(grades)

keys = grades.keys()
values = grades.values()
values = grades.items()

"Tim" in keys
"Tim" in grades

word_counts = {}

document = ["home", "work", "home"]


for word in document:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1
## if문
for word in document:
    try:
        word_counts[word] += 1
    except KeyError:
        word_counts[word] = 1
## 에러 제외
for word in document:
    previous_count = word_counts.get(word, 0)
    word_counts[word] = previous_count + 1

## 각 단어의 개수를 새는 법


from collections import defaultdict
for word in document:
    word_counts[word] += 1


dd_list = defaultdict(list)
dd_list[2].append(1)

dd_dict = defaultdict(dict)
dd_dict["Joel"]["City"] = "Seattle"

dd_pair = defaultdict(lambda: [0,0])
dd_pair[2][1] = 1


from collections import Counter
c = Counter([0,1,2,0])
## 개수 새주는 라이브러리

word_counts = Counter(document) ## 단어 개수도 새줌

for word, count in word_counts.most_common(10):
    print(word, count)
## 자주나온 단어 10개순으로 출력

primes_below_10 = {2,3,5,7}

s = set()
s.add(1)
s.add(2)
s.add(2)
x = len(s)
y = 2 in s
z = 3 in s
## set은 집합 == 중복원소 허용 안함

##stopwords_list = ["a", "an", "at"] + hundreds_of_other_words + ["yet", "you"]




if 1 > 2:
    message = "if only 1 were greater than two"
elif 1 > 3:
    message = "elif stands for 'else if'"
else:
    message = "when all else fails use else (if you want to)"
print(message)
## 3번출력


parity = "even" if 3 % 2 == 0 else "odd"
print(parity)

x = 0
while x < 10:
    print("f{x} is less than 10")
    x += 1

for x in range(10):    
    print("f{x} is less than 10")

for x in range(10):
    if x == 3:
        continue
    if x == 5:
        break
    print(x)
## conttinue 이번 루프 스킵, break 루프 종료

one_is_less_than_two = 1 < 2
true_equals_false = True == False

x = None
assert x == None, "this is the not the Pyhonic way to check for None"
assert x is None, "this is the Pyhonic way to check for None"

s = lambda :"asdf"
if s:
    first_char = s[0]
else:
    first_char = ""

first_char = s and s[0]

safe_x = x or 0

safe_x = x if x is not None else 0
## x가 0이 아니라면 x를 넣음

all([True, 1 , {3}]) ## 거짓이 없어야 하므로 true
all([True, 1 , {}]) ## 빈칸은 false
any([True, 1 , {}]) ## 참이 하나라도 있으므로 true
all([]) # 거짓이 없으므로 true
any([]) # 참이 없으므로 false

x = [4,1,2,3]
y = sorted(x)
x.sort() ## 오름차순 정렬
x = sorted([-4,1,-2,3], key=abs, reverse = True)  #데이터를 절대값하여 내림차순으로 정렬



even_numbers = [x for x in range(5) if x % 2 == 0] ## 0~4사이의 수에서 짝수를 리스트에 넣는다.
squares = [x * x for x in range(5)] ## 0~4 사이의 수를 제곱

## 이는 dict나 set에도 적용 가능
## 역시 _도 불필요한값으로 치워줌

pairs = [(x,y) for x in range(10) for y in range (5)]
pairs = [(x,y) for x in range(10) for y in range (x+1, 5)]
## for를 연속해서 붙이기도 가능

assert 1 + 1 == 2
assert 1 + 1 == 2, "1 + 1 should equal 2 but didn't"


def smallest_item(xs):
    return min(xs)

assert smallest_item([10,20,5,40]) == 5 ## 결과가 true가 아니라면 에러가 뜸



class CountingClicker:
    """함수처럼 클래스에도 주석 추가가능"""
    def __init__(self, count = 0): ##생성자
        self.count = count
    def __repr__(self, count = 0): ##반환
        return "CountingClicker(count={self.count})"
    def click(self, num_times = 1):
        self.count += num_times
    def read(self):
        return self.count
    def reset(self):
        self.count = 0


clicker1 = CountingClicker()
clicker1 = CountingClicker(100)
clicker1 = CountingClicker(count=100)

clicker = CountingClicker()
assert clicker.read() == 0, "clicker should start with count 0"
clicker.click()
clicker.click()
clicker.click()

assert clicker.read() == 2. "after"
clicker.click()
assert clicker.read() == 0, "after2"

