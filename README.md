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
