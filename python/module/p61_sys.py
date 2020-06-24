import sys
print(sys.path)
## 시스템 패스 걸려있는 폴더는 그냥 가져올수있음

from test_import import p62_import

p62_import.sum2()

from test_import.p62_import import sum2

sum2()