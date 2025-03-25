# Hands on machine learning
<img src="https://github.com/SeungMin2001/Hands-On_Machine-Learning_Study/blob/main/hands_on_machinelearning_img.png"  width="500" height="700">

> "Hands-On Machine Learning with Scikit-Learn Keras & Tensorflow" 의 대한 공부 기록."  
> — Aurelien Geron, 《Hands-On Machine Learning with Scikit-Learn Keras & Tensorflow 3판》, 2023
> 2장은 전반적인 내용을 다루므로 ReadMe에 저장, 나머지 장은 파일로 각각 저장한다.

[2장-code](https://colab.research.google.com/github/rickiepark/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb#scrollTo=E4Js1hcvoyh6)
<br>

[3장 분류](https://github.com/SeungMin2001/Hands-On_Machine-Learning_Study/tree/main/03.Classification)



<br>

## 02_end_to_end_machine_learning_project.
### 데이터 가져오기
```py
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
```
<br>

### 데이터 구조 훑어 보기

```py
housing.head()
housing.info()
housing["occean_proximity"].value_counts()
housing.describe()

housing.hist(bins=50, fizsize=(12,8))
plt.show()
```
<br>

### 테스트 세트 만들기
- 보통 20% 를 테스트 세트로 만듬
- shuffle_end_split_data() 를 사용한 테스트세트 만들기
```py
import numpy as np

def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = shuffle_end_split_data(housing, 0.2)
np.random.seed(42)
```
<br>

- sklearn - train_tesdt_split 를 사용한 테스트세트 만들기
```py
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```
<br>

- 계층적 샘플링
```
지금까지는 수순한 랜덤 샘플링 방식이였다.
지금부터는 계층적 샘플링에 대해 공부해보자.
```

```py
import numpy as np
import pandas as pd

housing["income_cat"] = pd.cut(housing["median_income"],   # 연속형 데이터
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],  # 구간 설정
                               labels=[1, 2, 3, 4, 5])  # 각 구간의 레이블


# housing 에 "income_cat"이라는 새로운 속성을 만들고 그 열에 pd.cut을 사용해서 만든 데이터를 넣어주는 과정
# median_income 를 기준으로 범주형 데이터를 만들어주고 새로운 속성값에다가 넣어주는 과정
# pd.cut() : 인자로 (데이터,bins,label) 를 넣어준다.
#       bins에 구간을 넣어주면 자동으로 이상,미만 기준이 생긴다. ex: 1,3,5 면 1이상 3미만, 3이상 5 미만 이라는 기준에 따라 범주형 데이터로 분류된다.

housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()

# 여기서 sort_index() 를 안해주면 값이 큰 애부터 내림차순으로 출력됨. 해주면 인덱스 순서대로 1,2,3,4,5 로 출력

```
<br>

- StratifiedShuffleSplit 를 사용한 데이터 나누기
```py
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42) # n_splits=10 이므로 10번의 쪼개기가 실행(2,8 비율로 나눠짐)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

# 여기서 StratifiedShuffleSplit.split()는 인덱스를 반환하기 때문에 iloc로 접근해야하고 행 인덱스기 때문에 행 전체를 가져오게 된다. 즉 가로줄
```
<br>

- 테스트 세트에서 소득 카테고리의 비율
```py
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
```
<br>

- 여기서 train_set, test_set 둘다 value.count() 분포가 거의 같다는것을 볼수 있는데, 이건 계층적 샘플링이 됬다는거고 StratifiedShuffleSplit이 알아서 해줬다는 뜻이 된다.
- 이 예제에서는 income_cat특성을 다시 사용하지 않으므로 이 열을 삭제하도록 한다.
```py
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```
<br>

### 데이터 이해를 위한 탐색과 시각화
- 지리적 데이터 시각화
```py
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2) #grid=격자,alpha=투명도
plt.show()
```
<br>

```py
# 원의 반지름(인구수) = s , 색상(가격) = c , 컬러맵 = cmap (jet 사용)
housing.plot.scatter(x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, figsize=(10, 7))
plt.show()
```
<br>

### 상관관계 조사하기
- 보통 피어슨의 상관계수를 활용해서 구한다. 1=비례함수, 0=상관관계 없음, -1=음의 비례함수
```py
corr_matrix = housing.corr(numeric_only=True)
# 판다스 2.0 버전에서 기본값이 False로 저절로 바뀜, 그래서 numeric_only=True로 지정해줘야함
# numeric_only = True 안해주면 오류발생. 범주형 변수때문에. 

corr_matrix["median_house_value"].sort_values(ascending=False)
# corr_matrix에서 median_house_value 를 기준으로 한 상관관계를 정렬해서 출력(asending=True:오룸차순, asendinf=False:내림차순)
# ascending 기본값은 False 즉 내림차순 이므로 오름차순 하고싶을때만 asceding=True 값을 넣어주면 된다.
```
<br>

- 여러 산점도를 보여주는 scatter_matrix
```py
# scatter_matrix 함수를 이용하여 여러개의 상관관계 산점도를 그릴수 있음
# 상관관계를 수치적으로 미리 구한 corr_matrix를 사용하지 않는이유 : scatter_matrix함수가 housing 안에 있는 데이터들의 산점도를 나타내주면 시각적으로 상관관계를 볼수 있게 되는것
# corr_matrix를 사용해서 시각화 하려면 seaborn-heatmap을 사용해야함. scatter_matrix로는 corr_matrix를 사용한 시각화를 못함
from pandas.plotting import scatter_matrix
attributes=["median_house_value", "median_inocme", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attribute], figsize(12,8))
plt.show()
```
<br>
<b>scatter_matrix 결과를 보면 median_income,median_house_value 이 두개의 상관관계가 강해보인다는것을 알수 있다 (중요!)</b>
<br><b>단 선형적인 관계에서의 상관계수 관계만을 봐야함. 비선형적인 관계의 데이터들에서 상관관계를 분석하면 안됨</b>
<br>

### 특정 조합으로 데이터들의 인사이트 얻기
- ex): 가구강 방 개수 = 전체 방 / 전체 가구
```py
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)

# 이렇게 조합을 통해 새로운 변수를 만들고 상관관계를 구해보면 훨씬 유의미한 상관관계가 나타날수 있다.
```
<br>

### 머신러닝 알고리즘을 위한 데이터 준비
```py
# 이전에 계층적 샘플링 해놓은 strat_train_set에서 median_house_value 열을 제거한 상태로 housing에 대입, 즉 X,Y 분리과정
# 분리한 Y는 housing_labels에 저장 (지도학습 이기 때문에 Y(종속변수) 를 따로 저장)
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```
<br>

- 데이터 정제 (옵션 3개가 있음)
    1. 해당 구역을 제거
    2. 전체 특성을 삭제
    3. 누락된 값을 어떤 값으로 채움(0,평균,중간값 등) 이를 Imputation(대체)
```py
housing.dropna(subset=["total_bedrooms"], inplace=True)    # 옵션 1

housing.drop("total_bedrooms", axis=1)                     # 옵션 2

median = housing["total_bedrooms"].median()                # 옵션 3
housing["total_bedrooms"].fillna(median, inplace=True)

```
<br>

- 데이터를 최대한 유지하기 위해 옵션3을 선택, sklearn 에 SimpleImputer 를 사용
```py
from sklearn.impute import SimpleImputer
imputer =  SimpleImputer(strategy="median") # 중간값으로 채워준다.
```

- 범주형 데이터를 제외한 숫자형 데이터들만 선택해서 housing_num에 넣어주는 과정
```py
housing_num = housing.select_dtypes(include=[np.number])
```
<br>

- fit을 통해 미리 지정해둔 median 값으로 결측치를 대체하는 과정
```py
# 이 데이터세트에는 total_bedrooms에만 결측치가 있지만, 새로운 데이터셋이 들어왔을때 어느 컬림에 결측치가 있는지 파악이 힘드므로 모든 컬럼에 대해 imputer.fit을 해준다.(단 수치형만)
imputer.fit(housing_num)
```
- 마지막으로 계산된 통계값(중앙값 등) 을 저장 및 출력
```py
imputer.statistics_
# = housing_num.median().values 같은 결과를 나타내는 코드

X=imputer.transform(housing_num)
# transform은 데이터 수정이 아닌 새로운 배열을 반환함!! 중요
# transform 을 실행하려면 반드시 fit 이 선행되야함!! 중요


# 즉 fit을 통해 중앙값을 계산, 그리고 statistics 속성에 저장. 단 housing_num을 변형하진 읺음
# transform 을 실행하면 statistics 에 저장된 값으로 "대체" 하는 과정!
```
<br>

- 텍스트와 범주형 특성 다루기
```py
# 보통 변수명에 cat이 있으면 카테고리의 약자이므로 범주형 데이터를 가르킴
# housing["ocean_proximity"] = 1차원 = Series
# housing[["ocean_proximity"]] = 2차원 = FataFrame
```
<br>

- 범주형 데이터 (텍스트 데이터->숫자 데이터) 변환방법 : OrdinalEncoder
```py
# <주의!> 범주형 데이터간의 순서,크기가 없다면 쓰면 안됨. 모델이 숫자의 크기/순서 관계를 학습해버릴수 있기 때문!
# ex: red=0, blue=1, green=2 일때 green>blue>red 이런식의 관계로 잘못 학습할수 있음! -> OneHotEncoder 을 사용해야함

from skelarn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat) #.fit_transform을 통한 변환

ordinal_encoder.categories_  #categories_ 인스턴스 변수를 사용해 카테고리 리스트를 얻을수 있음
```
<br>

- 범주형 데이터 (텍스트 데이터->숫자 데이터) 변환방법 : OneHotEncoder
```py
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# 희소행렬로 저장, numpy배열로 바꿀려면 toarray() 를 사용
housing_cat_1hot.toarray()

# sparse=False, sparse_output=False 로 해주면 transform() 메서드가 numpy 배열을 반환하도록 바꿀수 있음

# 범주형 -> 원-핫 여기서 하나의 이진특성으로 표현하는 방법
df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
pd.get_dummies(df_test)

# 왠만하면 oneHotEncoding .fit_transform 을 통해 범주형 데이터를 다루는게 좋다/
```
<br>

- 특정 스케일링과 변환. (데이터를 일정한 범위로 변환하는 과정)
<br>

- MinMaxScaler = Normalization(정규화)
```py
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxscaler(feture_range(-1,1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num) #fit_transform 을 여기서도 써줘야 값이 바뀜
```
<br>

- StandardScaler = Standardization(표준화)
```py
# 희소행렬을 밀집행렬로 안바꾸고 스케일링 하고싶을때 표준화를 사용할때 with_mean=False를 해주면 평균을 내지 않고 표준편차로 나누기만 해서 가능하다.
# 표준화를 하면 평균이 항상 0, 표준편차는 항상 1이 된다
# 표준화는 이상치에 영향을 덜 받는다.
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
```
<br>

- 꼬리가 두꺼운 특성을 처리하는 방법 -> Bucketizing(버킷타이징:구간화).  (스케일링으로는 한계가 있음)
- 특정한 범위로 나누어 범주형 데이터로 변환하는 방법 -> 버킷타이징
- housing["housing_median_age"] 특성이 멀티모달 분포를 띄고있음(mode, 즉 정점이 두개 이상인 분포)
- 멀티모달 분포 해결, 멱법칙 분포 -> 1.버킷타이징     2.방사 기저 함수(RBF)
- 원-핫 인코딩 전에 버킷라이징을 해주고 실행할수 있음, 버킷라이징(수치형->범주형) -> 원-핫 인코딩(범주형->이진벡터)
<br>

- 방사 기저 함수(RBF) = 거리 기반 함수
```py
from sklearn.metrics.pairwise import rbf_kernel

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
# 즉 이 함수는 유사도를 측정한다. 얼마나 비슷한 값인가를 측정한다는 뜻이다.
# gamma 값이 클수록 커널이 민감하게 반응한다, 값이 작을수록 더 넓은 범위의 데이터에 영향을 미칩니다.
# housing_median_age 와 35와의 유사도를 age_simil_35 에 넣어주는 코드이다.
# 예를 들어 10,20,30,40 데이터가 있을때 35와의 유사도는 30,40이 가장 클것이고 10이 가장 낮은 유사도를 가질것이다.
```
<br>

- 스케일링을 한 데이터로 결과가 나올때 원본 데이터로 결과를 보기 위한 역변환
- sklearn.compose - TransformedTargetRegressor 사용
```py
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)
```
<br>

- 사용자 정의 변환기
- FunctionTransFormer 을 활용
```py
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

# FunctionTransFormer 인자값으로 데이터 변환을 수행할 함수, inverse_func=np.exp(지수함수) 라는 역변환 여부를 넣어준다.
# housing["population"] 에 로그 변환을 적용한 코드 (log_transformer.transform()을 사용해서 변화)
```
<br>

- rbf_kernel 과 FunctionTransFormer 을 활용한 사용자 정의 변환
```py
# 아직 이해하기 힘든 개념들이 있기 때문에 스킵

rbf_transformer = FunctionTransformer(rbf_kernel,
                                      kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

#
```
<br>

- 변환 파이프라인
```py
# fit_transform()을 해줘야 데이터가 변환됨, 안하면 그냥 파이프라인 구조만 나타낼수 있음.
from sklearnpipeline import Pipeline
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),  # ->결측값 처리
    ("standardize", StandardScaler()),   # -> 표준화 : 평균=0 표준편차=1로 변환한다.
])

# + fit, transform, fit+transform 의 차이점.
# fit(): 학습만 수행
# transform(): 이미 학습된 데이터를 변환만 수행
# fit_transform(): 학습하고 그 값을 변환함.
# 파이프라인은 fit_transform() 메서드를 가져야함
```
<br>

- make_pipeline 사용하기
```py
# make_pipeline 이 이름을 자동으로 만들어줌(ex:SimpleImputer->simpleimputer)
from sklearn.pipeline import make_pipeline
num_pipeline(SimpleImputer(strategy="median"), StandarScaler())
# Pipeline 과의 차이점은 이름을 안지어준다는점 하나다.
# 만약 같은 이름이 있다면 뒤에 숫자를 붙혀서 구별한다.
```
<br>

- housing_num을 파이프라인에 넣어보기
```py
housing_num_prepared = num_pipeline.fit_transform(housing_num)
housing_num_prepared[:2].round(2)

# 데이터프레임으로 재구성 하려면 get_feature_names_out() 을 사용
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index) # ->housing_num의 인덱스를 그대로 사용한다는 뜻 
# get_feature_names_out : 변환된 특성 이름을 반환한다.
# 예를들면 sstandardScaler를 사용하여 데이터의 각 열이 변환된 후, 각 특성에 대한 새로운 이름을 반환한다.

# 1.get_feature_names_out() 으로 변환된 데이터 열들을 얻고 저장시킨다.
# 2. 얻은 새로운 열들을 dataframe() 에 인자로 같이 넘겨준다.
# 3. 넘파이 배열에서 데이터프레임 배열로 변환될때 인자로 넘겨준 컬럼인자로 새롭게 업데이트 된다.
```
<br>

- ColumnTransform 사용하기
- 각 영에 대해 서로 다른 변환을 동시에 적용할 수 있도록 도와주는 역할을 함
```py
from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([     # 각각 다른 파이프라인을 다른열에 한번에 적용할수 있음
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
```
<br>

- make_column_selector, make_column_trasform 사용하기
```py
from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer( # 이름 자동으로 만들어줌(pipeline1,pipeline2 이렇게 만들어짐
    (num_pipeline, make_column_selector(dtype_include=np.number)), # np.number:수치형 데이터
    (cat_pipeline, make_column_selector(dtype_include=object)), # object:범주형데이터,문자열
)
# 이렇게 각각의 파이프라인을 지정한 열에 적용할수 있음. 여기서 좋은점은 여러개의 열과 파이프라인을 한번에 매칭해서 실행할수 있다는 점임.
```
<br>

- 내가 만든 파이프라인 적용하기
```py
housing_prepared=preprocessing.fit_transform(housing)
# 여기서도 numpy배열로 반환되지만 dataframe 데이터로 반환할수 있음
# preprocessing.get_feature_names_out() 을 사용해서
```
<br>

### 모델 선택과 훈련

- 훈련 세트에서 훈련하고 평가하기
```py
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

# 만들어놓은 파이프라인을 실행하고 그 다음 LinearRegression 을 실행하게 하는 코드
# make_pipeline을 실행하면 preprocessing -> LinearRegression 을 하게 설정되고
# fit을 돌려주면 학습을 하게 된다.
```
<br>

### 교차 검증으로 평가하기 (이 부분부터는 책을 다 읽고 다시 돌아와서 학습하자)
### 모세 미세 튜닝
- 최적의 하이퍼파라미터 조합 찾기
### 론칭, 모니터링, 시스템 유지 보수








