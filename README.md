# Hands-On_Machine-Learning_Study
### Useing California Housing Prices datasets
##### Data downloading
```py
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path=Path("datasets/housing.tgz") #경로 객체 생성
    if not tarball_path.is_file(): # tarball_path == empty 이면 실행
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url="https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrive(url, tarball_path) #urlretrieve() 로 tarball_path에 파일저장
        with tarfile.open(tarball_path) as housing_tarball: #with문으로 close 해줌
            housing_tarball.extractcall(path="datasets") #받아온 파일을 housing_tarball에 저장후 datasets파일에 extract 해줌
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing=load_housing_data()
```
### Classification
##### Mnist
```py
from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784',as_frame=False)
mnist.keys()
```
```py
X,y=mnist.data,mnist.target
```
```py
import matplotlib.pyplot as plt

def plot_digit(image_data): # X안에 있는 데이터를 28,28로 reshape 해줘야 함 픽셀 데이터이기 때문
    image=image_data.reshape(28,28) 
    plt.imshow(image,cmap="binary") #cmap변수를 안넣어줘도 되고 gray로 넣어줘도 되고 binary로 넣어줘도 됨 색깔차이임, binary가 젤 보기 좋음
    plt.axis("off")

for i in range(10):
    plot_digit(X[i])
    plt.show()
```
```py
X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:] #mnist 데이터셋은 훈련세트 60000개, 테스트세트 10000개로 이미 나눠져 있는 데이터셋임
```
