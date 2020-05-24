# 통계적 가설검정
# 가설이란 '이 동전은 앞뒤가 나올 확률이 공평한 동전이다.' 등과 같은 주장
# 기존의 입장을 귀무가설 이와 대비되는 입장을 대립가설(H1)

from typing import Tuple
import math

def normal_approximation_to_binomial(n: int, p: float):
    """Binomial(n, p)에 해당되는 mu(평균)와 sigma(표준편차) 계산"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

from scratch.probability import normal_cdf

# 누적 분포 함수는 확률변수가 특정 값보다 작을 확률을 나타냄
normal_probability_below = normal_cdf

# 만약 확률 변수가 특정 값보다 작지 않다면, 특정 값보다 크다는 것을 의미함
def normal_probability_above(lo: float, mu: float = 0, sigma: float = 1):
    """mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo보다 클 확률"""
    return 1 - normal_cdf(lo, mu, sigma)

# 만약 확률변수가 hi보다 작고 lo보다 작지 않다면 확률변수는 hi와 lo 사이에 존재함
def normal_probability_between(lo: float, hi: float, mu: float = 0, sigma: float = 1):
    """mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo와 hi 사이에 있을 확률"""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# 만약 확률변수가 범위 밖에 존재한다면 범위 안에 존재하지 않는다는 것을 의미
def normal_probability_outside(lo: float, hi: float, mu: float = 0, sigma: float = 1):
    """mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo와 hi 사이에 없을 확률"""
    return 1 - normal_probability_between(lo, hi, mu, sigma)




from scratch.probability import inverse_normal_cdf

def normal_upper_bound(probability: float, mu: float = 0, sigma: float = 1):
    """P(Z <= z) = probability인 z값을 반환"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float, mu: float = 0, sigma: float = 1):
    """P(Z <= z) = probability인 z값을 반환"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float, mu: float = 0, sigma: float = 1):
    """
    입력한 probability 값을 포함하고
    평균을 중심으로 대칭적 구간을 반환
    """
    tail_probability = (1 - probability) / 2

    # 구간의 상한은 tail_probability 값 이상의 확률 값을 갖고 있다.
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # 구간의 하한은 tail_probability 값 이하의 확률 값을 갖고 있다.
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

lower_boun, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0) # (469, 531)





# p = 0.5라고 가정할 때, 유의수준이 5%인 구간
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# p = 0.55인 경우의 실제 평균과 표준편차
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# 제 2종 오류란 귀무가설을 기각하지 못한다는 의미
# 즉, X가 주어진 구간 안에 존재할 경우를 의미
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability


hi = normal_upper_bound(0.95, mu_0, sigma_0)
# 결과값은 526 (< 531, 분포 상위 부분에 더 높은 확률을 주기 위해)

type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability


# p-value
# 귀무가설이 참이라고 가정하고 실제로 관측된 값보다 더 극단적인 값이 나올 확률

def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1):
    """
    mu(평균)와 sigma(표준편차)를 따르는 정규분포에서
    x같이 극단적인 값이 나올 확률은 얼마나?
    """
    if x >= mu:
        # 만약 x가 평균보다 크다면 x보다 큰 부분이 꼬리
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # 만약 x가 평균보다 작다면 x보다 작은 부분이 꼬리
        return 2 * normal_probability_below(x, mu, sigma)

two_sided_p_value(529.5, mu_0, sigma_0)

# 시뮬레이션
import random

extreme_value_count = 0
for _ in range(1000):
    num_heads = sum(1 if random.random() < 0.5 else 0 # 앞면이 나온 경우
                    for _ in range(1000)) # 동전을 1000번
    if num_heads >= 530 or num_heads <= 470: # 극한 값
        extreme_value_count += 1  # 몇 번 나오는지 세어봄

# p-value was 0.062 => ~62 extreme values out of 1000
assert 59 < extreme_value_count < 65, "{extreme_value_count}"

# 계산된 p-value가 5%보다 크기 때문에 귀무가설을 기각안함

two_sided_p_value(531.5, mu_0, sigma_0)

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

# 동전의 앞면이 525번 나왔다면 단측검정을 위한 p-value는 다음과 같이 계산되며
# 귀무가설을 기각하지 않을 것이다
upper_p_value(524.5, mu_0, sigma_0) # 0.061

# 만약 동전의 앞면이 527번 나왔다면 p-value는 다음과 같이 계산되며
# 귀무가설을 기각할 것이다
upper_p_value(526.5, mu_0, sigma_0) # 0.047


# 신뢰구간
#math.sqrt(p * (1 - p) / 1000)

# 동전을 1,000번 던져서 앞면이 525번 나온 경우
p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000) # 0.0158
normal_two_sided_bounds(0.95, mu, sigma) # [0.4940, 0.5560]

# 동전을 1,000번 던져서 앞면이 540번 나온 경우
p_hat = 540 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000) # 0.0158
normal_two_sided_bounds(0.95, mu, sigma) # [0.5091, 0.5709]


# p 해킹
from typing import List

def run_experiment():
    # 동전을 1000번 던져서 True = 앞, False = 뒷면
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment: List[bool]):
    # 신뢰구간을 5%로 설정
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

random.seed(0)
experiments = [run_experiment() for _ in range(1000)]
num_rejections = len([experiment
                      for experiment in experiments
                      if reject_fairness(experiment)])

assert num_rejections == 46

# 주어진 데이터에 대해 다양한 가설들을 검정하다 보면 이중 하나는 반드시 의미 있는 가설로 보일 수 있다
# 이걸 데이터를 잘 만지면 p값이 낮게 만들 수 있다. 그러면 안됨
# 가설은 데이터를 보기 전에 세운다
# 데이터를 전처리 할 때는 세워둔 가설을 잠시 잊는다
# p-value가 전부는 아니다(대안으로 베이즈 추론을 사용할 수 있음)


# 예시: A/B test 해보기 P 105쪽
# 광고 A와 광고 B가 있다
# 광고 A를 본 1000명 중 990명이 광고를 클릭
# 광고 B를 본 1000명 중 10명이 광고를 클릭
# 이러한 명확한 차이가 없다면 통계적 추론을 통해 인사이트를 얻어야 한다


def estimated_parameters(N: int, n: int):
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

# pA와 pB가 같다는(즉, pA - pB = 0) 귀무가설은 다음의 통계치로 검정 가능
def a_b_test_statistics(N_A: int, n_A: int, N_B: int, n_B: int):
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

# 위 식은 대략 표준정규분포를 따름

z = a_b_test_statistics(1000, 200, 1000, 180) # -1.14
two_sided_p_value(z) # 0.254

z = a_b_test_statistics(1000, 200, 1000, 150) # -2.94
two_sided_p_value(z) # 0.003


# 베이즈 추론
# 알려지지 않은 파라미터를 확률변수로 보는 방법
def B(alphaL: float, beta: float):
    """모든 확률값의 합이 1이 되도록 해주는 정규화 값"""
    return math.gamma(alpah) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x: float, alpha: float, beta: float):
    if x <= 0 or x >= 1:        # [0, 1] 구간 밖에서는 밀도가 없다
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)
