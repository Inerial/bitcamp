from matplotlib import pyplot as plt

#선 그래프
years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

plt.plot(years, gdp, color = 'green',marker = 'o', linestyle = 'solid')
# x 연도, y gdp인 선그래프

plt.title("Nominal GDP") #제목

plt.ylabel("Billions of $") #y축 제목
plt.show()

#막대그래프
movies = ["Annie Hall", "Ben_Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

plt.bar(range(len(movies)), num_oscars) # 각 막대 크기를 num_oscars로 지정

plt.title("My Favorite Movies")    
plt.ylabel("# of Academy Awards") 

plt.xticks(range(len(movies)), movies) # 각 막대의 이름을 movies로 지정
plt.show()

#히스토그램
from collections import Counter

grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

histogram = Counter(min(grade // 10 * 10, 90) for grade in grades) # 10점 단위로 묶음

plt.bar([x + 5 for x in histogram.keys()], histogram.values(), 10, edgecolor = (0, 0, 0))


plt.axis([-5, 105, 0, 5])       


plt.xticks([10 * i for i in range(11)])    
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()


mentions = [500, 505]
years = [2017, 2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("# of times I heard somone say 'data science'")

plt.ticklabel_format(useOffset = False)
# 이렇게 하지 않으면 matplotlib이 x축에 0, 1 레이블을 달고
# 주변부 어딘가에 +2.013e3이라고 표기해 둘 것임(나쁜 matplotlib)

plt.axis([2016.5, 2018.5, 499, 506])
plt.title("Look at the 'Huge' Increase!")
plt.show()
# 오해를 불러일으키는 y축은 500 이상의 부분만 보여줄 것이다

plt.axis([2016.5, 2018.5, 0, 550])
plt.title("Not So Huge Anymore")
plt.show()
# 오해를 불러일으키지 않는 그래프




varience = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error = [x + y for x, y in zip(varience, bias_squared)]
xs = [i for i, _ in enumerate(varience)]

# 한 차트에 여러 개의 선을 그리기 위해
# plt.plot을 여러 번 호출할 수 있다
plt.plot(xs, varience, 'g-', label = 'varience') # 실선
plt.plot(xs, bias_squared, 'r-.', label = 'bias^2') # 일점쇄선
plt.plot(xs, total_error, 'b:', label = 'total error') # 점선

# 각 선에 레이블을 미리 달아놨기 때문에
# 범례(legend)를 쉽게 그릴 수 있다.
plt.legend(loc = 9)
plt.xlabel("model complexity")
plt.xticks([])
plt.title("The Bias-Varience Tradeoff")
plt.show()



#산점도
friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)


for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label,
        xy = (friend_count, minute_count),
        xytext = (5, -5), #약간 공간주기
        textcoords = 'offset points')
    
plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
plt.show()

# 변수들끼리 비교할 때 matplotlib이 자동으로 축의 범위를 설정하게 하면 공정한 비교 못할수 있음
test_1_grades = [99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter(test_1_grades, test_2_grades)
plt.title("Axes Aren't Comparable")
plt.xlabel("test 1 grade")
plt.ylabel("test 2 grade")
plt.show()