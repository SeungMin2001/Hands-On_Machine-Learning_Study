# Hands on machine learning
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
