import seaborn as sns

# 타이타닉 데이터셋을 불러옵니다.
titanic = sns.load_dataset('titanic')

# 데이터프레임의 첫 5행 출력
print(titanic.head())

# 기본 통계 확인
print(titanic.describe())

# 결측치 확인
print(titanic.isnull().sum())

# describe() 함수에서 확인할 수 있는 통계 항목들에 대한 설명
# count: 각 feature에서 유효한 (결측치가 아닌) 값의 개수를 보여줍니다.
# std: 표준편차(standard deviation)는 데이터의 분포가 평균으로부터 얼마나 퍼져 있는지를 나타냅니다.
# min: 해당 feature의 최소값을 보여줍니다.
# 25%: 1사분위수(25th percentile)는 데이터의 하위 25%가 포함되는 값입니다.
# 50%: 2사분위수(중앙값, 50th percentile)는 데이터의 중간값을 나타냅니다.
# 75%: 3사분위수(75th percentile)는 데이터의 상위 25%가 포함되는 값입니다.
# max: 해당 feature의 최대값을 보여줍니다.