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
