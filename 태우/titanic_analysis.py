import seaborn as sns  # seaborn 라이브러리를 불러옵니다.
from sklearn.model_selection import train_test_split  # 데이터 분할을 위한 라이브러리
from sklearn.preprocessing import StandardScaler  # 데이터 스케일링을 위한 라이브러리

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

# 6. 결측치 처리
# - Age(나이)의 결측치는 중앙값으로 대체합니다.
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# - Embarked(승선 항구)의 결측치는 최빈값으로 대체합니다.
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# 7. 수치형으로 인코딩
# - 성별(Sex)을 숫자로 변환합니다: 남자는 0, 여자는 1
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})

# - 생존 여부(alive)를 숫자로 변환합니다: 살아있으면 1, 죽었으면 0
titanic['alive'] = titanic['survived'].map({0: 0, 1: 1})

# - 승선 항구(Embarked)를 숫자로 변환합니다: 'C'는 0, 'Q'는 1, 'S'는 2
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# 8. 새로운 feature 생성: 가족 크기
# - SibSp(타이타닉호에 동승한 자매 및 배우자의 수)와 Parch(부모 및 자식의 수)를 합쳐 가족 크기를 계산합니다.
# - 본인을 포함하기 위해 1을 더합니다.
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

# 9. 최종 결과 확인
# - 생성된 feature들을 출력하여 확인합니다.
print(titanic[['sex', 'alive', 'embarked', 'family_size']].head())

# 10. 모델 학습을 위한 데이터 준비
# - 필요한 feature와 target을 선택합니다.
titanic = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size']]
X = titanic.drop('survived', axis=1)  # feature
y = titanic['survived']  # target

# 11. 데이터 스케일링
scaler = StandardScaler()  # StandardScaler 객체 생성
X_scaled = scaler.fit_transform(X)  # 데이터를 스케일링합니다.

# 12. 학습 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 13. 결과 확인
print("학습 데이터 형태:", X_train.shape)
print("테스트 데이터 형태:", X_test.shape)