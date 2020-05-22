users = [
    {"id":0, "name":"Here"},
    {"id":1, "name":"Dunn"},
    {"id":2, "name":"Sue"},
    {"id":3, "name":"Chi"},
    {"id":4, "name":"Thor"},
    {"id":5, "name":"Clive"},
    {"id":6, "name":"Hicks"},
    {"id":7, "name":"Devin"},
    {"id":8, "name":"Kate"},
    {"id":9, "name":"Klein"}
]

friendship_pairs = [(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),(6,8),(7,8),(8,9)]

friendships = {user["id"] : [] for user in users}
## friendships 안에  {id : 빈리스트} 형태의 딕셔너리를 id개수만큼 순서대로 삽입

for i, j, in friendship_pairs:
    friendships[i].append(j)
    friendships[j].append(i)

## friendships 안의 딕셔너리 안의 빈 리스트에 연결되어있는 값을 넣어줌
## 예를들어 (0,1)이면 id = 0 인 key값을 가진 딕셔너리에 빈 리스트에 1을 넣어주고
## id = 1 인 key값을 가진 딕셔너리에 빈 리스트에 0을 넣어준다

def number_of_friends(user):
    """user의 친구는 몇 명일까?"""
    user_id = user["id"]
    friend_ids = friendships[user_id]
    return len(friend_ids)
## 딕셔너리 형태의 데이터 user를 매개변수로 받아 id값을 꺼내 friendships에 저장되어있는 
## 친구리스트를 꺼내 데이터의 길이로 친구수를 리턴함

total_connections = sum(number_of_friends(user) for user in users)
## 위의 함수를 users 의 딕셔너리 수만큼 시행하여 모조리 더해준다
## 친구리스트의 개수의 총 합 == 연결고리의 개수

num_users = len(users) # 유저의 수
avg_connections = total_connections / num_users # 각 유저의 평균 연결고리 수

num_friends_by_id = [(user["id"], number_of_friends(user)) for user in users]
## (user_id, user 친구수) 의 형식의 토플을 유저수만큼 생성해 만든 리스트

num_friends_by_id.sort(
    key=lambda id_and_friends: id_and_friends[1], reverse=True
)

## 각 라인에 직접 뭘 넣는다는데 잘 모르겠다. 대충 토플의 1 index에 있는 number of friends 값을 기준으로 정렬하는것 같다. (reverse = True : 큰수부터)

def foaf_ids_bad(user):
    return [foaf_id for friend_id in friendships[user["id"]] for foaf_id in friendships[friend_id]]

##받은 매개변수 user
## user 안의 id값의 위치를 key값으로 friendships 리스트에서 value값인 list 가져옴
## 그 list들은 user의 친구들 => 그 리스트를 다시 friendships 리스트에 입력하여 친구들의 친구들을 가져옴

print(friendships[0])
print(friendships[1])
print(friendships[2])

## 각각 유저의 친구리스트 출력

from collections import Counter
## 글자 개수 세주는 함수

def friends_of_friends(user):
    user_id = user["id"]
    return Counter(
        foaf_id for friend_id in friendships[user_id] for foaf_id in friendships[friend_id] if foaf_id != user_id and foaf_id not in friendships[user_id]
    )
## 사용자의 친구는 제외한 친구의 친구

print(friends_of_friends(users[3]))
## Counter({0: 2, 5: 1}) == 0이 2개 5가 한개

interests = [
    (0,"Hadoop"),(0,"Big Data"),(0,"HBase")
]
## 대충 이것만 침

def data_scientists_who_like(target_interest):