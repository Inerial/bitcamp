#3. 딕셔너리 # 중복 x
# {키 : 밸류}

a = {1 : 'hi', 2: 'hello'}
print(a) # {1:'hi', 2:'hello'}
print(a[1])  # hi

b = {'hi' : 1, 'hello' : 2}
print(b['hello']) # 3

del a[1] # {2:'hello'}
print(a)
del a[2] # {}
print(a)

a = {1:'a', 2: 'b', 3 : 'c'}
print(a) # {1:'a', 2: 'b', 3:'c'}

b = {1:'a', 2: 'a', 3 : 'a'}
print(b) # {1:'a', 2:'a', 3:'a'}
# key는 중복이 되지 않지만 value는 중복되도 상관이 없다.

a = {'name': 'yun', 'phone' : '010', 'birth' : '0511'}
print(a.keys()) # dict_keys(['name','phone','birth'])
print(a.values()) # dict_values(['yun','010','0511'])
print(type(a)) # <class 'dict'>
print(a.get('name')) # 'yun'
print(a['name']) # yun
print(a.get('phone')) # '0511'
print(a['phone']) # '0511


